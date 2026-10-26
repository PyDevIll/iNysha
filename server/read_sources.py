"""What the agent actually read — and where a fact came from.

Facts about the outside world are only as good as their source. This module
records which files and URLs the tools really returned, so
``memory_remember(source=...)`` can demand a citation that exists: a URL the
model invented is simply not in the table, and a file it never opened cannot be
quoted.

Three rules keep it cheap and honest:

  * recording happens **after** a successful tool result, never before — a
    failed read is not a source;
  * one upsert per read (~0.1 ms, WAL), no tokens added to any prompt;
  * a citation stays valid for ``MEMORY_SOURCE_WINDOW_DAYS`` (7 by default).
    Older rows are kept for diagnostics but cannot be cited fresh.

``exec_shell``/``exec_python`` are not tracked on purpose: after ``cat`` there
is no way to know what was read. If the agent wants to cite a file, it reads it
with ``fs_read``; otherwise the fact is stored as its own, without a source.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from loguru import logger

from server import config as app_config
from server.memory_store import MemoryStore, now_ts

# Tool argument that names the thing that was read.
FILE_TOOLS = {
    "fs_read": "file",
    "fs_read_docx": "file",
    "fs_stat": "path",
    "fs_tail": "file",
}
URL_TOOLS = {
    "tavily_browse": "url",
    "vision_analyze_url": "image_url",
}
# These return a page full of links; the snippets were in context, so citing a
# URL from the result is legitimate.
SEARCH_TOOLS = {"tavily_search", "tavily_map", "tavily_research"}
TRACKED_TOOLS = set(FILE_TOOLS) | set(URL_TOOLS) | SEARCH_TOOLS

MAX_SOURCES_PER_CALL = 10
_MAX_AGE_DAYS = 3650

_URL_RE = re.compile(r"https?://[^\s<>\"'`)\]]+")
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "yclid")
_FAILURE_PREFIXES = (
    "error", "not found", "file not found", "not a ", "ошибка", "не найдено",
    "path not found", "permission denied",
)
_TRAILING = ".,;:!?»)]}"


def normalize_url(raw: str) -> Optional[str]:
    """Canonical form for comparison: https, host as-is, no fragment/UTM."""
    try:
        parsed = urlsplit(raw.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query = urlencode([
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_PARAMS)
    ])
    # Both http and https canonicalise to https: the same page cited as http
    # would otherwise look like an unread source.
    return urlunsplit(("https", netloc, path, query, ""))


def canonical_ref(raw: str) -> Optional[tuple[str, str]]:
    """``("web"|"file", canonical)`` or ``None`` when it is not a source."""
    value = (raw or "").strip().strip("<>\"'`")
    if not value:
        return None
    lower = value.lower()
    if lower.startswith(("http://", "https://")):
        url = normalize_url(value)
        return ("web", url) if url else None
    if lower.startswith("file://"):
        value = value[7:]
    elif re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return None  # ftp://, ssh:// — не прочитанный источник
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or "/" in value or "\\" in value:
        return "file", str(candidate.resolve())
    # A bare word is not a path; do not turn "server" into a citation.
    return None


def _is_failure(text: str) -> bool:
    head = text.lstrip().lower()[:40]
    return any(head.startswith(prefix) for prefix in _FAILURE_PREFIXES)


def _file_meta(ref: str) -> tuple[Optional[int], Optional[int]]:
    """``(size, mtime_ns)`` — nanoseconds catch sub-second edits."""
    try:
        stat = os.stat(ref)
    except OSError:
        return None, None
    return int(stat.st_size), int(stat.st_mtime_ns)


def record_tool_read(
    store: MemoryStore,
    tool_name: str,
    args: dict,
    result,
    chat_id: Optional[str] = None,
) -> int:
    """Remember a successful read; returns how many sources were recorded."""
    if tool_name not in TRACKED_TOOLS:
        return 0
    text = str(result or "")
    if not text.strip() or _is_failure(text):
        return 0

    refs: list[tuple[str, str]] = []
    arg_key = FILE_TOOLS.get(tool_name) or URL_TOOLS.get(tool_name)
    if arg_key:
        canonical = canonical_ref(str((args or {}).get(arg_key) or ""))
        if canonical is not None:
            refs.append(canonical)
    if tool_name in SEARCH_TOOLS:
        for url in _URL_RE.findall(text)[:MAX_SOURCES_PER_CALL]:
            canonical = canonical_ref(url.rstrip(_TRAILING))
            if canonical is not None:
                refs.append(canonical)

    # Одна и та же ссылка в выдаче встречается трижды; записываем один раз.
    refs = list(dict.fromkeys(refs))
    recorded = 0
    for kind, ref in refs:
        size = mtime = None
        if kind == "file":
            size, mtime = _file_meta(ref)
        if store.record_source(
            kind=kind, ref=ref, chat_id=chat_id, size=size, mtime=mtime
        ):
            recorded += 1
    if recorded:
        logger.debug(
            f"memory: tool source(s) recorded tool={tool_name} "
            f"count={recorded} refs={[ref for _kind, ref in refs][:3]}"
        )
    return recorded


def age_days(source: dict) -> int:
    try:
        seen = int(source.get("last_seen") or 0)
    except (TypeError, ValueError):
        return _MAX_AGE_DAYS
    return max(0, (now_ts() - seen) // 86400)


def resolve_source(
    store: MemoryStore,
    raw: str,
    window_days: Optional[int] = None,
) -> Optional[dict]:
    """A recent, actually-read source for this citation, or ``None``."""
    canonical = canonical_ref(raw)
    if canonical is None:
        return None
    kind, ref = canonical
    row = store.get_source(kind, ref)
    if row is None:
        return None
    window = (window_days if window_days is not None
              else app_config.MEMORY_SOURCE_WINDOW_DAYS) * 86400
    if now_ts() - int(row["last_seen"]) > window:
        logger.debug(
            f"memory: source too old for citation kind={kind} ref={ref[:100]}"
        )
        return None
    return {
        "kind": kind,
        "ref": ref,
        "title": row["title"],
        "last_seen": int(row["last_seen"]),
        "size": row["size"],
        "mtime": row["mtime"],
    }


def _short_path(ref: str) -> str:
    path = Path(ref)
    for base in (Path(app_config.ROOT), Path(app_config.DATA_DIR)):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return path.name or ref


def label(source: Optional[dict]) -> str:
    """Short prompt label: ``[web example.com, 3 дн.]`` / ``[file data/x.md]``."""
    if not source:
        return ""
    kind = source.get("kind")
    ref = str(source.get("ref") or "")
    if kind == "web":
        host = (urlsplit(ref).hostname or ref)[:40]
        return f"[web {host}, {age_days(source)} дн.]"
    if kind == "file":
        return f"[file {_short_path(ref)}]"
    return ""

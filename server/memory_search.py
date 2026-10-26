"""Read-only search over the episodic memory.

One implementation for both callers: the ``memory_search`` tool (the agent
asking on purpose) and, later, the automatic block in ``build_llm_messages``.
Keeping them on the same function is what makes the "measure it before you
plug it in" promise practical — the eval script exercises exactly the path
production will use.

Queries go through ``server.memory_text.fts_query``: lemmas first (so
"сервером" finds "сервер"), and a prefix retry when lemmas find nothing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger

from server import config as app_config
from server.memory_store import MemoryStore
from server.memory_text import DEFAULT_PREFIX_LEN, fts_query, preview

# One hit is a preview of the utterance, not the whole transcript: episode
# texts can be thousands of characters and the tool result lands in context.
SNIPPET_CHARS = 260
DEFAULT_K = 8


@dataclass
class EpisodeHit:
    id: int
    ts: int
    speaker: str
    chat_id: str
    text: str
    score: float

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.ts).strftime("%d.%m.%Y %H:%M")

    def as_line(self, index: int) -> str:
        who = {"human": "человек", "agent": "я", "tool": "tool",
               "system": "system"}.get(self.speaker, self.speaker)
        return f"{index}. [{self.when}] {who}: {preview(self.text, SNIPPET_CHARS)}"


def _raw_search(store: MemoryStore, match: str, limit: int) -> list[tuple[int, float]]:
    """FTS match → (rowid, score) in relevance order, limited BEFORE the join.

    An unbounded ``JOIN episodes`` on a common word was measured at 80 ms;
    ranking and slicing inside the FTS table first keeps it at ~1 ms.
    """
    rows = store.connection.execute(
        "SELECT rowid, bm25(episodes_fts) AS score FROM episodes_fts "
        "WHERE episodes_fts MATCH ? ORDER BY score LIMIT ?",
        (match, limit),
    ).fetchall()
    # bm25 is negative and "better" means more negative; expose a positive
    # relevance so callers can reason about thresholds without sign tricks.
    return [(int(row["rowid"]), -float(row["score"])) for row in rows]


def _fetch(store: MemoryStore, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = store.connection.execute(
        f"SELECT id, ts, speaker, chat_id, text FROM episodes "
        f"WHERE id IN ({placeholders})",
        tuple(ids),
    ).fetchall()
    return {
        int(row["id"]): {
            "ts": int(row["ts"]),
            "speaker": str(row["speaker"]),
            "chat_id": str(row["chat_id"]),
            "text": str(row["text"]),
        }
        for row in rows
    }


def search_episodes(
    store: MemoryStore,
    query: str,
    k: int = DEFAULT_K,
    chat_id: Optional[str] = None,
    min_score: Optional[float] = None,
) -> list[EpisodeHit]:
    """Return up to ``k`` episodes for ``query``, best first.

    ``min_score`` drops weak matches (relevance is positive, higher = better),
    which is how the automatic prompt block will avoid injecting noise.
    """
    k = max(1, min(int(k), 50))
    match = fts_query(query)
    if not match:
        return []

    started = time.perf_counter()
    hits: list[EpisodeHit] = []
    used_prefix = False
    # Over-fetch: chat filtering and min_score can throw items away.
    for attempt in (0, 1):
        pairs = _raw_search(store, match, k * 4)
        rows = _fetch(store, [row_id for row_id, _score in pairs])
        hits = []
        for row_id, score in pairs:
            row = rows.get(row_id)
            if row is None:
                continue
            if chat_id is not None and row["chat_id"] != str(chat_id):
                continue
            if min_score is not None and score < min_score:
                continue
            hits.append(EpisodeHit(
                id=row_id,
                ts=row["ts"],
                speaker=row["speaker"],
                chat_id=row["chat_id"],
                text=row["text"],
                score=score,
            ))
            if len(hits) >= k:
                break
        if hits or attempt == 1:
            break
        # No hits with lemmas: retry with truncated stems ("переза"*).
        prefix_match = fts_query(query, prefix_len=DEFAULT_PREFIX_LEN)
        if not prefix_match or prefix_match == match:
            break
        used_prefix = True
        match = prefix_match

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.debug(
        f"memory: search q={preview(query, 40)!r} k={k} hits={len(hits)} "
        f"prefix_fallback={used_prefix} chat={chat_id or '-'} "
        f"{elapsed_ms:.2f}ms"
    )
    return hits


@dataclass
class NodeHit:
    id: int
    kind: str
    name: str
    text: str
    score: float
    # Where the fact came from: provenance is what lets the block mark external
    # knowledge and hide a fact whose file has changed since it was read.
    origin: str = "agent"
    confidence: float = 0.0
    source: Optional[dict] = None

    def as_line(self, index: int) -> str:
        return f"{index}. [{self.kind}] {self.name}: {preview(self.text, SNIPPET_CHARS)}"


def _node_payload(row) -> dict:
    try:
        payload = json.loads(row["data"] or "{}")
    except (ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _node_text(row) -> str:
    """Node body: the ``text`` field of its JSON payload, else the raw data."""
    payload = _node_payload(row)
    return str(payload.get("text") or payload.get("note") or "")


def search_nodes(
    store: MemoryStore,
    query: str,
    k: int = DEFAULT_K,
    scope: Optional[str] = None,
) -> list[NodeHit]:
    """Search remembered facts (``nodes``), newest-and-best first."""
    k = max(1, min(int(k), 50))
    match = fts_query(query)
    if not match:
        return []
    pairs = store.connection.execute(
        "SELECT rowid, bm25(nodes_fts) AS score FROM nodes_fts "
        "WHERE nodes_fts MATCH ? ORDER BY score LIMIT ?",
        (match, k * 4),
    ).fetchall()
    hits: list[NodeHit] = []
    for pair in pairs:
        row = store.get_node(int(pair["rowid"]))
        if row is None or row["status"] == "retired":
            continue
        if scope is not None and row["scope"] not in (scope, "shared"):
            continue
        payload = _node_payload(row)
        source = payload.get("source")
        hits.append(NodeHit(
            id=int(row["id"]),
            kind=str(row["kind"]),
            name=str(row["name"]),
            text=str(payload.get("text") or payload.get("note") or ""),
            score=-float(pair["score"]),
            origin=str(row["origin"] or "agent"),
            confidence=float(row["confidence"]),
            source=source if isinstance(source, dict) else None,
        ))
        if len(hits) >= k:
            break
    logger.debug(f"memory: node search q={preview(query, 40)!r} hits={len(hits)}")
    return hits


def format_all(
    query: str,
    nodes: list[NodeHit],
    episodes: list[EpisodeHit],
    note: str = "",
) -> str:
    """Facts first (they are short and vetted), then transcript fragments."""
    if not nodes and not episodes:
        return f"По запросу «{preview(query, 80)}» ничего не найдено."
    lines: list[str] = []
    budget = app_config.MEMORY_BLOCK_TOKENS
    used = 0

    def append(line: str) -> bool:
        nonlocal used
        cost = max(1, len(line) // 3)
        if used + cost > budget:
            return False
        lines.append(line)
        used += cost
        return True

    header = f"Найдено: фактов {len(nodes)}, фрагментов {len(episodes)}"
    append(f"{header} (запрос: «{preview(query, 80)}»{note}):")
    if nodes:
        append("Из памяти (факты):")
        for index, node in enumerate(nodes, start=1):
            if not append(node.as_line(index)):
                break
    if episodes:
        append("Из истории:")
        for index, hit in enumerate(episodes, start=1):
            if not append(hit.as_line(index)):
                append("… обрезано лимитом памяти")
                break
    return "\n".join(lines)


def format_hits(query: str, hits: list[EpisodeHit], note: str = "") -> str:
    """Render hits for a tool result, within the memory token budget."""
    if not hits:
        return f"По запросу «{preview(query, 80)}» ничего не найдено."
    lines = [f"Найдено {len(hits)} (запрос: «{preview(query, 80)}»{note}):"]
    used = 0
    budget = app_config.MEMORY_BLOCK_TOKENS
    for index, hit in enumerate(hits, start=1):
        line = hit.as_line(index)
        # Same char/3 estimate the context manager falls back to; exact
        # tiktoken counting here would only add a dependency edge.
        cost = max(1, len(line) // 3)
        if used + cost > budget and index > 1:
            lines.append(f"… ещё {len(hits) - index + 1} (лимит {budget} токенов)")
            break
        lines.append(line)
        used += cost
    return "\n".join(lines)

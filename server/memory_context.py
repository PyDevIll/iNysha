"""The memory block that goes into the prompt once per turn.

Three constraints shape this module (see ``data/memory_schema_draft.md``):

  * **prefix cache** — the block is placed at the tail, right before the first
    message of the current turn, and its content is fixed for the whole turn
    (``TurnMemory``), so the 15 possible LLM calls of one turn share one block
    and the cached prefix is not torn on every iteration;
  * **budget** — ``MEMORY_BLOCK_TOKENS`` in tokens, not "as much as fits";
  * **optional** — everything here is off unless ``MEMORY_BACKEND=sqlite``,
    and any failure yields an empty block, never an exception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from server import config as app_config
from server.memory_search import NodeHit, EpisodeHit, search_episodes, search_nodes
from server.memory_store import MemoryStore
from server.memory_text import preview

HEADER = "## Из памяти"
_MAX_ITEMS = None  # бюджет в токенах решает; количество — производное


def enabled() -> bool:
    """Whether the agent prompt gets a memory block at all."""
    return (app_config.MEMORY_BACKEND or "json").strip().lower() == "sqlite"


def warmup() -> bool:
    """Load lemmatisation dictionaries before the first turn (idempotent)."""
    from server.memory_text import warmup as _warmup

    ok = _warmup()
    logger.info(f"memory: warmup morphology={'ok' if ok else 'degraded'}")
    return ok


def _cost(text: str) -> int:
    # Same char/3 estimate the context manager falls back to; keeping it here
    # avoids importing context_manager (which imports this module's siblings).
    return max(1, len(text) // 3)


def render_block(
    nodes: list[NodeHit],
    episodes: list[EpisodeHit],
    budget_tokens: int,
) -> str:
    """Render selected memory within ``budget_tokens``; empty when nothing fits."""
    if not nodes and not episodes:
        return ""
    lines = [HEADER]
    used = _cost(HEADER)

    def add(line: str) -> bool:
        nonlocal used
        cost = _cost(line)
        if used + cost > budget_tokens:
            return False
        lines.append(line)
        used += cost
        return True

    if nodes:
        lines.append("Факты:")
        used += _cost("Факты:")
        for node in nodes:
            if not add(f"- [{node.kind}] {node.name}: {preview(node.text, 180)}"):
                break
    if episodes:
        lines.append("Из истории:")
        used += _cost("Из истории:")
        for hit in episodes:
            who = "человек" if hit.speaker == "human" else "я"
            if not add(f"- [{hit.when[:10]}] {who}: {preview(hit.text, 160)}"):
                break

    if len(lines) <= 2:  # заголовок и ничего больше
        return ""
    return "\n".join(lines)


def _dedupe(episodes: list[EpisodeHit]) -> list[EpisodeHit]:
    """Drop near-identical fragments (the same reply often appears twice).

    Memory contains retried sends and repeated summaries; two hits starting
    with the same 120 normalised characters are the same memory to the model.
    """
    seen: set[str] = set()
    unique: list[EpisodeHit] = []
    for hit in episodes:
        key = " ".join(hit.text.split()).lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def build_block(
    store: MemoryStore,
    query: str,
    chat_id: Optional[str] = None,
    budget_tokens: Optional[int] = None,
) -> str:
    """Search and render one block; never raises."""
    if not query.strip():
        return ""
    budget = budget_tokens or app_config.MEMORY_BLOCK_TOKENS
    started = time.perf_counter()
    try:
        nodes = search_nodes(store, query, k=app_config.MEMORY_BLOCK_MAX_ITEMS)
        episodes = _dedupe(search_episodes(
            store,
            query,
            k=app_config.MEMORY_BLOCK_MAX_ITEMS,
            chat_id=chat_id,
        ))
        block = render_block(nodes, episodes, budget)
        record_usage(store, nodes, episodes)
    except Exception as e:  # память не имеет права ломать ход
        logger.warning(f"memory: block build failed, injecting nothing: {e}")
        return ""
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.debug(
        f"memory: block q={preview(query, 40)!r} chat={chat_id or '-'} "
        f"nodes={len(nodes)} episodes={len(episodes)} "
        f"tokens~{_cost(block)}/{budget} {elapsed_ms:.2f}ms"
    )
    return block


def record_usage(
    store: MemoryStore,
    nodes: list[NodeHit],
    episodes: list[EpisodeHit],
) -> None:
    """Remember what was actually served — the only way to judge usefulness."""
    rows = [("node", node.id) for node in nodes]
    rows += [("episode", hit.id) for hit in episodes]
    if not rows:
        return
    now = int(time.time())
    try:
        store.connection.executemany(
            "INSERT INTO memory_index(item_type, item_id, last_used, hits) "
            "VALUES(?,?,?,1) ON CONFLICT(item_type, item_id) DO UPDATE SET "
            "last_used=excluded.last_used, hits=hits+1",
            [(item_type, item_id, now) for item_type, item_id in rows],
        )
        store.connection.commit()
    except Exception as e:  # pragma: no cover - метрика, не функциональность
        logger.debug(f"memory: usage recording failed: {e}")


@dataclass
class TurnMemory:
    """Per-turn memo of the block.

    ``Agent`` resets it when a new input arrives; every ``llm_request`` of that
    turn then reuses the same string at the same position, which is what keeps
    the prompt prefix cacheable.
    """

    last_key: Optional[tuple[str, Optional[str]]] = None
    block: str = ""
    calls: int = 0

    def reset(self) -> None:
        self.last_key = None
        self.block = ""
        self.calls = 0

    def get(
        self,
        query: str,
        chat_id: Optional[str],
        build: "callable",
    ) -> str:
        key = (query, chat_id)
        if self.last_key == key:
            self.calls += 1
            logger.trace(f"memory: block cache hit (call {self.calls})")
            return self.block
        self.last_key = key
        self.calls = 1
        self.block = build()
        return self.block

"""Memory tools for the agent: search now, explicit writes next.

Search is read-only and local (SQLite + FTS5, ~1 ms), so it is safe as a direct
call even in a voice turn — it does not block on the network and does not
change the world. Writes (``memory_remember`` / ``memory_forget``) are a
separate stage and deliberately not here yet.
"""

from __future__ import annotations

import time

from loguru import logger

from server.memory_search import DEFAULT_K, format_hits, search_episodes
from server.memory_store import get_store


async def memory_search(query: str, k: int = DEFAULT_K, chat_id: str = "") -> str:
    """Search long-term memory (past conversations) for earlier utterances.

    Use when the answer depends on something said earlier: past decisions,
    keys, credentials locations, project details, preferences, "what did we
    agree about X". Returns dated snippets with who said it.
    """
    if not (query or "").strip():
        return "Пустой запрос: нечего искать."
    store = get_store()
    started = time.perf_counter()
    hits = search_episodes(
        store,
        query,
        k=k,
        chat_id=chat_id.strip() or None,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.debug(
        f"memory_search tool: q={query[:60]!r} k={k} hits={len(hits)} "
        f"chat={chat_id or '-'} {elapsed_ms:.2f}ms"
    )
    return format_hits(query, hits)


TOOL_DEFINITIONS = [
    (
        "memory_search",
        memory_search,
        "Search long-term memory of past conversations. Use when the answer "
        "depends on something said earlier (decisions, keys, projects, "
        "preferences, history of a chat). Read-only.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Что искать: ключевые слова, имена, темы",
                },
                "k": {
                    "type": "integer",
                    "description": "Сколько фрагментов вернуть (по умолчанию 8, максимум 50)",
                },
                "chat_id": {
                    "type": "string",
                    "description": "Ограничить поиск одним чатом (опционально)",
                },
            },
            "required": ["query"],
        },
    ),
]


def register_all(registry) -> None:
    registered = 0
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
        registered += 1
    logger.info(f"Registered {registered} memory tools")

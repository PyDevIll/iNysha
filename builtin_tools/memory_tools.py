"""Memory tools for the agent: search (read) and explicit remember/forget.

Search is local SQLite/FTS5 (~1-3 ms, read-only), so it is allowlisted for
direct voice use. Writes are deliberately **not** allowlisted: in a voice turn
a misheard phrase must go through ``start_tool`` (a separate turn), and a wrong
fact in memory is re-injected into every future prompt.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from server.memory_search import (
    DEFAULT_K,
    format_all,
    search_episodes,
    search_nodes,
)
from server.memory_store import get_store
from server.memory_write import (
    MemoryWriteError,
    current_trust,
    forget,
    remember,
)

MAX_K = 50


async def memory_search(query: str, k: int = DEFAULT_K, chat_id: str = "") -> str:
    """Search long-term memory: remembered facts and past conversation.

    Use when the answer depends on something said or decided earlier:
    past decisions, where keys live, project details, preferences, history of
    a chat. Returns short dated fragments with who said it.
    """
    if not (query or "").strip():
        return "Пустой запрос: нечего искать."
    store = get_store()
    started = time.perf_counter()
    nodes = search_nodes(store, query, k=max(1, min(int(k), MAX_K)))
    episodes = search_episodes(
        store,
        query,
        k=max(1, min(int(k), MAX_K)),
        chat_id=chat_id.strip() or None,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.debug(
        f"memory_search tool: q={query[:60]!r} k={k} nodes={len(nodes)} "
        f"episodes={len(episodes)} chat={chat_id or '-'} {elapsed_ms:.2f}ms"
    )
    return format_all(query, nodes, episodes)


async def memory_remember(
    kind: str,
    name: str,
    text: str,
    scope: str = "shared",
) -> str:
    """Remember one fact on purpose (survives restarts, comes back into context).

    Use only when the user says to remember something, or when a durable fact
    is clearly established. Never store secrets: keep where the key lives, not
    the key. Never store jokes or your own sarcasm as facts.
    """
    trusted, chat_id = current_trust()
    try:
        result = remember(
            get_store(),
            kind=kind,
            name=name,
            text=text,
            scope=scope,
            trusted=trusted,
            chat_id=chat_id,
        )
    except MemoryWriteError as e:
        logger.info(f"memory_remember rejected: {e}")
        return f"Не записано: {e}"
    verdict = "Запомнила" if result.created else "Обновила"
    origin = "оператор" if trusted else "сказано мной"
    logger.info(
        f"memory_remember: {verdict} [{result.kind}] {result.name!r} "
        f"id={result.node_id} scope={result.scope} ({origin})"
    )
    return (
        f"{verdict}: [{result.kind}] {result.name} "
        f"(id={result.node_id}, scope={result.scope}, источник: {origin})."
    )


async def memory_forget(
    node_id: Optional[int] = None,
    name: Optional[str] = None,
    kind: Optional[str] = None,
) -> str:
    """Forget a remembered fact (soft delete: kept in the database, hidden).

    Use when the user says to forget something, or when a fact is proven wrong.
    Nothing is physically deleted: the row is marked retired and its live
    relations are closed.
    """
    try:
        retired = forget(get_store(), node_id=node_id, name=name, kind=kind)
    except MemoryWriteError as e:
        logger.info(f"memory_forget rejected: {e}")
        return f"Не забыто: {e}"
    listing = ", ".join(f"{node_kind}/{node_name} (id={node_id_value})"
                        for node_id_value, node_kind, node_name in retired)
    logger.info(f"memory_forget: retired {len(retired)} → {listing}")
    return f"Забыто ({len(retired)}): {listing}. Строки сохранены, помечены retired."


TOOL_DEFINITIONS = [
    (
        "memory_search",
        memory_search,
        "Search long-term memory: remembered facts and past conversations. Use "
        "when the answer depends on something said or decided earlier "
        "(decisions, where keys live, projects, preferences, chat history). "
        "Read-only.",
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
                    "description": "Ограничить историю одним чатом (опционально)",
                },
            },
            "required": ["query"],
        },
    ),
    (
        "memory_remember",
        memory_remember,
        "Remember one durable fact on purpose. Only when asked to remember or "
        "when a fact is clearly established. Never secrets (store where the key "
        "lives, not the key), never jokes or sarcasm as facts.",
        {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["person", "project", "server", "rule", "fact",
                             "topic", "document"],
                    "description": "Тип понятия: person/project/server/rule/fact/topic/document",
                },
                "name": {
                    "type": "string",
                    "description": "Короткое имя факта, например 'оператор Андрей' или 'лимит ElevenLabs'",
                },
                "text": {
                    "type": "string",
                    "description": "Сам факт в одном-двух предложениях (до 2000 символов)",
                },
                "scope": {
                    "type": "string",
                    "enum": ["shared", "chat"],
                    "description": "shared — для всех чатов, chat — только текущий",
                },
            },
            "required": ["kind", "name", "text"],
        },
    ),
    (
        "memory_forget",
        memory_forget,
        "Forget a remembered fact (soft delete). Use when the user asks to "
        "forget or when a fact is proven wrong.",
        {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "id факта из memory_search",
                },
                "name": {
                    "type": "string",
                    "description": "Имя факта, если id неизвестен",
                },
                "kind": {
                    "type": "string",
                    "description": "Уточнить тип при поиске по имени",
                },
            },
        },
    ),
]


def register_all(registry) -> None:
    registered = 0
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
        registered += 1
    logger.info(f"Registered {registered} memory tools")

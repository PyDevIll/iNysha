"""Показать, что память подставит в промпт по конкретному запросу.

Отвечает на вопрос «почему этот факт не поднялся / почему поднялся мусор» без
ожидания хода и без порчи метрик: база копируется в /tmp, `memory_index` не
трогается, живой файл открывается только на чтение.

    .venv/bin/python scripts/memory_show.py "какой лимит у озвучки"
    .venv/bin/python scripts/memory_show.py --items "как шарить по гитхабу"
    .venv/bin/python scripts/memory_show.py --chat 326963375 --budget 200 "звук"

С `--items` печатается разбор: score, происхождение, размер и причина, по
которой элемент попал или не попал в блок (бюджет, квота внешних, файл
изменился). Ровно тот же код выбора и рендера, что в проде.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from server import config as app_config
from server import memory_context
from server.memory_store import MemoryStore
from server.memory_text import preview

REASONS = {
    "ok": "в блоке",
    "budget": "не влезло в бюджет",
    "external_quota": "квота внешних",
    "file_changed": "файл изменился",
}


def snapshot(db_path: Path, directory: str) -> Path:
    """Consistent copy of a live database (safe while the service writes)."""
    target = Path(directory) / "memory-snapshot.db"
    source = sqlite3.connect(
        f"file:{quote(str(db_path.resolve()))}?mode=ro", uri=True
    )
    try:
        copy = sqlite3.connect(target)
        try:
            source.backup(copy)
        finally:
            copy.close()
    finally:
        source.close()
    return target


def explain(
    db_path: Path, query: str, chat_id: str = "", budget: int | None = None
) -> dict:
    """Run the production selection on a throwaway copy of the database."""
    with tempfile.TemporaryDirectory(prefix="memory-show-") as tmp:
        copy_path = snapshot(db_path, tmp)
        store = MemoryStore(copy_path)
        try:
            return memory_context.explain_block(
                store, query, chat_id=chat_id or None, budget_tokens=budget
            )
        finally:
            store.close()


def _node_meta(node) -> str:
    return (
        f"score={node.score:.2f} {node.origin} conf={node.confidence:.2f} "
        f"{len(node.text)} симв."
    )


def _episode_meta(hit) -> str:
    return f"score={hit.score:.2f} {hit.when[:10]} {hit.speaker} {len(hit.text)} симв."


def format_report(result: dict, show_items: bool = False) -> str:
    lines = [
        f"запрос: «{preview(result['query'], 90)}»"
        + (f"  чат: {result['chat_id']}" if result.get("chat_id") else ""),
        f"бюджет: {result['budget']} токенов, занято ~{result['tokens']} "
        f"({result['elapsed_ms']:.1f} мс)",
    ]
    nodes = {node.id: node for node in result["nodes"]}
    episodes = {hit.id: hit for hit in result["episodes"]}

    if show_items:
        trace = result["trace"]
        if not trace:
            lines.append("ничего не найдено")
        for index, entry in enumerate(trace, start=1):
            ident = entry["kind"]
            if entry["kind"] == "node":
                node = nodes.get(entry["id"])
                detail = _node_meta(node) if node else ""
            else:
                hit = episodes.get(entry["id"])
                detail = _episode_meta(hit) if hit else ""
            reason = REASONS.get(entry["reason"], entry["reason"])
            lines.append(
                f"{index:>2}. {ident} #{entry['id']}  {detail}  → {reason}"
            )
        total = len(result["nodes"]) + len(result["episodes"])
        shown = sum(1 for entry in trace if entry["reason"] == "ok")
        if total > len(trace):
            lines.append(f"    ещё {total - len(trace)} не проверялись (бюджет)")
        lines.append(f"итого в блоке: {shown} из {total}")

    lines.append("")
    lines.append(result["block"] or "(пусто — в промпт ничего не добавляется)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="фраза, как её написал бы пользователь")
    parser.add_argument("--db", default=str(app_config.MEMORY_DB_PATH))
    parser.add_argument("--chat", default="", help="chat_id для фильтра истории")
    parser.add_argument("--budget", type=int, default=None,
                        help="переопределить бюджет блока в токенах")
    parser.add_argument("--items", action="store_true",
                        help="показать разбор: score, происхождение, причина")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        print(f"нет базы: {db_path}")
        return 1
    # Скрипт печатает только отчёт: служебные DEBUG-строки памяти не нужны.
    logger.remove()
    memory_context.warmup()
    result = explain(db_path, args.query, args.chat, args.budget)
    print(format_report(result, show_items=args.items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

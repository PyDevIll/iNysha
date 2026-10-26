"""Измеряет качество поиска по памяти до того, как он попадёт в промпт.

Два режима, потому что это разные вопросы:

  --auto N    «Можно ли найти известный эпизод по его же словам?»
              берём N случайных эпизодов, строим запрос из 3 самых редких лемм
              и проверяем, попал ли этот эпизод в топ-k. Это регрессионный
              сторож: ломается индексация или лемматизация — число падает.

  --golden f  «Находит ли память то, что человек спрашивает своими словами?»
              JSONL: {"query": "...", "expect": ["подстрока", ...]}.
              Это единственный способ честно решить, хватает ли FTS без векторов.

    .venv/bin/python scripts/memory_eval.py --auto 50
    .venv/bin/python scripts/memory_eval.py --golden data/memory_golden.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import config as app_config
from server.memory_search import search_episodes
from server.memory_store import MemoryStore
from server.memory_text import lemma, tokenize


def _rare_lemmas(store: MemoryStore, episode_id: int, text: str) -> list[str]:
    """Pick the most distinctive lemmas of an episode (low document frequency)."""
    tokens = sorted({lemma(t) for t in tokenize(text)})
    tokens = [t for t in tokens if len(t) >= 4][:12]
    scored = []
    for token in tokens:
        count = store.connection.execute(
            "SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH ?",
            (f'"{token}"',),
        ).fetchone()[0]
        if count:
            scored.append((count, token))
    scored.sort()
    return [token for _count, token in scored[:3]]


def run_auto(store: MemoryStore, count: int, seed: int = 1) -> int:
    rows = store.connection.execute(
        "SELECT id, text FROM episodes WHERE length(text) > 120 "
        "ORDER BY id"
    ).fetchall()
    random.seed(seed)
    sample = random.sample(rows, min(count, len(rows)))
    ranks: list[int] = []
    latencies: list[float] = []
    misses: list[tuple[str, str]] = []
    for row in sample:
        query_terms = _rare_lemmas(store, int(row["id"]), str(row["text"]))
        if len(query_terms) < 2:
            continue
        query = " ".join(query_terms)
        started = time.perf_counter()
        hits = search_episodes(store, query, k=5)
        latencies.append((time.perf_counter() - started) * 1000)
        position = next(
            (i for i, hit in enumerate(hits, start=1) if hit.id == int(row["id"])),
            None,
        )
        if position is None:
            misses.append((query, str(row["text"])[:80]))
        else:
            ranks.append(position)
    return _report("auto (по словам самого эпизода)", len(sample), ranks, latencies, misses)


def run_golden(store: MemoryStore, path: Path, k: int = 5) -> int:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    ranks: list[int] = []
    latencies: list[float] = []
    misses: list[tuple[str, str]] = []
    for case in cases:
        query = case["query"]
        started = time.perf_counter()
        hits = search_episodes(store, query, k=k)
        latencies.append((time.perf_counter() - started) * 1000)
        position = None
        for index, hit in enumerate(hits, start=1):
            lowered = hit.text.lower()
            # OR-семантика: подстроки — маркеры темы, а не точная формулировка.
            # Автоматически судить семантику мы не умеем, поэтому метрика
            # называется «тема всплыла», а не «ответ найден».
            if any(expected.lower() in lowered for expected in case["expect"]):
                position = index
                break
        if position is None:
            misses.append((query, ", ".join(case["expect"])))
        else:
            ranks.append(position)
    return _report(f"golden {path.name}", len(cases), ranks, latencies, misses)


def _report(label: str, total: int, ranks: list[int], latencies: list[float],
            misses: list[tuple[str, str]]) -> int:
    found = len(ranks)
    recall = found / total if total else 0.0
    print(f"\n=== {label}")
    print(f"запросов: {total}, найдено: {found} ({recall:.0%})")
    for k in (1, 3, 5):
        hit_at = sum(1 for rank in ranks if rank <= k)
        print(f"  recall@{k}: {hit_at / total:.0%}" if total else "  —")
    if ranks:
        print(f"  MRR: {statistics.mean(1 / r for r in ranks):.3f}")
    if latencies:
        ordered = sorted(latencies)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        print(f"  latency p50/p95: {p50:.2f} / {p95:.2f} мс")
    if misses:
        print(f"\n  промахи ({len(misses)}):")
        for query, expected in misses[:15]:
            print(f"   · {query!r} → ожидалось {expected!r}")
    return 0 if recall >= 0.7 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(app_config.DATA_DIR))
    parser.add_argument("--db", default=None)
    parser.add_argument("--auto", type=int, default=0)
    parser.add_argument("--golden", default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    db_path = Path(args.db) if args.db else data_dir / "memory.db"
    if not db_path.exists():
        print(f"нет базы: {db_path}")
        return 1

    store = MemoryStore(db_path)
    try:
        if store.stats()["episodes"] == 0:
            print("в базе нет эпизодов — сначала scripts/import_history.py --apply")
            return 1
        code = 0
        if args.golden:
            golden = Path(args.golden)
            if not golden.is_absolute():
                golden = (data_dir / golden) if not golden.exists() else golden
            code |= run_golden(store, golden, args.k)
        if args.auto:
            code |= run_auto(store, args.auto)
        if not args.golden and not args.auto:
            parser.print_help()
    finally:
        store.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

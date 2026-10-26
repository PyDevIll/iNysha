"""Разбор кандидатов из `pending_facts`: принять, поправить или отклонить.

Экстрактор только предлагает. В граф факт попадает после явного решения —
поэтому здесь, а не в фоновом цикле.

    .venv/bin/python scripts/memory_pending.py --list
    .venv/bin/python scripts/memory_pending.py --accept 3
    .venv/bin/python scripts/memory_pending.py --accept 3 --name "лимит ElevenLabs"
    .venv/bin/python scripts/memory_pending.py --reject 4
    .venv/bin/python scripts/memory_pending.py --accept-all --min-confidence 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import config as app_config
from server.memory_store import MemoryStore, get_store
from server.memory_text import preview
from server.memory_write import MemoryWriteError, remember


def _payload(row) -> dict:
    try:
        data = json.loads(row["payload"])
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def _print(rows) -> None:
    if not rows:
        print("кандидатов нет")
        return
    print(f"{'id':>4}  {'conf':>4}  {'kind':<8} {'name':<34} текст")
    for row in rows:
        payload = _payload(row)
        print(
            f"{row['id']:>4}  {payload.get('confidence', 0):>4.2f}  "
            f"{payload.get('kind', '?'):<8} "
            f"{preview(payload.get('name', ''), 33):<34} "
            f"{preview(payload.get('text', ''), 80)}"
        )


def accept(
    store: MemoryStore,
    pending_id: int,
    name: str | None = None,
    text: str | None = None,
) -> int:
    """Approve one candidate; returns the created/updated node id."""
    row = store.connection.execute(
        "SELECT * FROM pending_facts WHERE id=? AND status='pending'",
        (int(pending_id),),
    ).fetchone()
    if row is None:
        raise MemoryWriteError(f"кандидат {pending_id} не найден или уже разобран")
    payload = _payload(row)
    result = remember(
        store,
        kind=str(payload.get("kind") or "fact"),
        name=name or str(payload.get("name") or ""),
        text=text or str(payload.get("text") or ""),
        chat_id=row["chat_id"],
        # Принято человеком вручную → это операторский факт, не догадка агента.
        trusted=True,
        confidence=max(0.7, float(payload.get("confidence") or 0.7)),
        origin="operator",
    )
    store.reviewed_pending(pending_id, result.node_id)
    return result.node_id


def reject(store: MemoryStore, pending_id: int) -> bool:
    return store.reviewed_pending(int(pending_id), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(app_config.DATA_DIR))
    parser.add_argument("--db", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--accept", type=int, default=None)
    parser.add_argument("--reject", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--accept-all", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    db_path = Path(args.db) if args.db else data_dir / "memory.db"
    if not db_path.exists():
        print(f"нет базы: {db_path}")
        return 1

    store = get_store(db_path)
    try:
        if args.accept is not None:
            node_id = accept(store, args.accept, args.name, args.text)
            print(f"принято: pending {args.accept} → node {node_id}")
            return 0
        if args.reject is not None:
            ok = reject(store, args.reject)
            print("отклонено" if ok else "не найдено или уже разобрано")
            return 0 if ok else 2
        if args.accept_all:
            accepted = 0
            skipped = 0
            for row in store.list_pending(limit=args.limit):
                payload = _payload(row)
                if float(payload.get("confidence") or 0) < args.min_confidence:
                    skipped += 1
                    continue
                try:
                    accept(store, int(row["id"]))
                    accepted += 1
                except MemoryWriteError as e:
                    print(f"  id={row['id']}: {e}")
                    skipped += 1
            print(f"принято: {accepted}, пропущено: {skipped}")
            return 0

        _print(store.list_pending(limit=args.limit))
        print("\nпринять: --accept ID | отклонить: --reject ID | всё: --accept-all")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""One manual extraction pass over real history (dry run by default).

The background extractor only sees *new* utterances: its cursor starts at the
newest episode, so importing history never triggers a hundred LLM calls.  This
script is the controlled exception — run it on a copy or with a small ``--limit``
to see what the extractor would produce.

    .venv/bin/python scripts/memory_extract.py --limit 8
    .venv/bin/python scripts/memory_extract.py --limit 8 --apply
    .venv/bin/python scripts/memory_extract.py --data-dir /tmp/copy --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import config as app_config
from server.memory_extract import (
    build_messages,
    make_client,
    parse_facts,
    select_batch,
    store_pending,
)
from server.memory_store import MemoryStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(app_config.DATA_DIR))
    parser.add_argument("--db", default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--cursor", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    db_path = Path(args.db) if args.db else data_dir / "memory.db"
    if not db_path.exists():
        print(f"нет базы: {db_path}")
        return 1

    store = MemoryStore(db_path)
    try:
        batch = select_batch(
            store, args.cursor, args.limit, app_config.MEMORY_EXTRACT_MIN_CHARS
        )
        if not batch.episodes:
            print("нет новых реплик для разбора")
            return 0
        print(f"реплик в батче: {len(batch.episodes)}, чаты: {batch.chats}")
        if args.show_prompt:
            for message in build_messages(batch):
                print(f"\n--- {message['role']}\n{message['content']}")
        try:
            client = make_client()
        except RuntimeError as e:
            print(f"клиент не создан: {e}")
            return 1

        response = asyncio.run(client.chat.completions.create(
            model=app_config.DEEPSEEK_MODEL,
            messages=build_messages(batch),
            stream=False,
            temperature=0.2,
            max_tokens=1200,
        ))
        reply = response.choices[0].message.content or ""
        facts = parse_facts(reply)
        print(f"\nкандидатов: {len(facts)}")
        print(json.dumps(facts, ensure_ascii=False, indent=2))

        if args.apply:
            inserted = store_pending(store, batch, facts)
            print(f"\nв pending_facts добавлено: {inserted}")
            print("принять в граф: пока вручную (см. data/memory_schema_draft.md)")
        else:
            print("\nпробный прогон; записать: --apply")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

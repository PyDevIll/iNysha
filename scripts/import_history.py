"""Import ``message_history.md`` into the episodes table.

The transcript on disk is already the agent's history; this makes it
searchable.  Parsing is deliberately the same everywhere:

  * blocks are separated by ``---``;
  * a block starts with ``**role:** [dd.mm.yyyy, HH:MM]``;
  * transport markers (``[Time: ...][MAX messenger | chat_id=...]``) are
    stripped from the text but used to recover ``chat_id``;
  * an assistant reply inherits the chat of the last user message, because
    only user turns carry the MAX marker in the file.

Re-import is safe: ``episodes`` is unique by ``(chat_id, ts, speaker, hash)``.

    .venv/bin/python scripts/import_history.py                  # dry run
    .venv/bin/python scripts/import_history.py --apply
    .venv/bin/python scripts/import_history.py --data-dir /tmp/copy
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import config as app_config
from server.memory_episodes import (
    extract_chat_id,
    parse_entry_time,
    resolve_db_path,
    strip_meta,
)
from server.memory_store import MemoryStore, episode_hash

HEADER_RE = re.compile(
    r"^\s*\*\*(?P<role>user|assistant|tool|system):\*\*\s*\[(?P<ts>[^\]]*)\]\s*\n?",
    re.IGNORECASE,
)
ROLE_TO_SPEAKER = {
    "user": "human",
    "assistant": "agent",
    "tool": "tool",
    "system": "system",
}
COMMIT_EVERY = 250


@dataclass
class ParsedEpisode:
    chat_id: str
    speaker: str
    kind: str
    ts: int
    text: str
    scope: str
    ref: Optional[str] = None

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.chat_id, self.ts, self.speaker, episode_hash(self.text))


def sent_message(body: str) -> Optional[tuple[str, str, str]]:
    """Recover ``(text, mid, chat_id)`` from a ``max_send_message`` tool result.

    Most of the agent's own replies never reached an assistant message: the
    text was delivered by the MAX tool, so ``message_history.md`` stores an
    empty assistant block and the payload in a tool block.  Without this the
    episodic layer would lose roughly two thirds of what the agent said.
    """
    payload = body.strip()
    if not payload.startswith("{"):
        return None
    try:
        data = ast.literal_eval(payload)
    except (ValueError, SyntaxError, MemoryError):
        return None
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if not isinstance(result, dict):
        return None
    sender = result.get("sender") or {}
    if not isinstance(sender, dict) or not sender.get("is_bot"):
        return None
    message = result.get("body") or {}
    if not isinstance(message, dict):
        return None
    text = (message.get("text") or "").strip()
    if not text:
        return None
    recipient = result.get("recipient") or {}
    chat_id = str(recipient.get("chat_id") or "") or "unknown"
    return text, str(message.get("mid") or ""), chat_id


def split_blocks(text: str) -> Iterator[str]:
    """Yield transcript blocks without the ``---`` separators."""
    for chunk in text.split("\n---\n"):
        block = chunk.strip()
        if block:
            yield block


def parse_history(
    text: str,
    include_tools: bool = False,
    include_sent: bool = True,
) -> Iterator[ParsedEpisode]:
    current_chat: Optional[str] = None
    for block in split_blocks(text):
        header = HEADER_RE.match(block)
        if not header:
            continue
        role = header.group("role").lower()
        speaker = ROLE_TO_SPEAKER[role]
        body = block[header.end():]

        found = extract_chat_id(body)
        if found:
            current_chat = found
        chat_id = found or current_chat or "unknown"

        ts = parse_entry_time(header.group("ts").strip())
        if ts is None:
            continue

        if speaker == "tool":
            sent = sent_message(strip_meta(body)) if include_sent else None
            if sent:
                text_sent, mid, sent_chat = sent
                if sent_chat != "unknown":
                    current_chat = sent_chat
                yield ParsedEpisode(
                    chat_id=sent_chat,
                    speaker="agent",
                    kind="message",
                    ts=ts,
                    text=text_sent,
                    scope=(
                        f"chat:{sent_chat}" if sent_chat != "unknown" else "shared"
                    ),
                    ref=mid or None,
                )
            if include_tools:
                raw = strip_meta(body)
                if raw:
                    yield ParsedEpisode(
                        chat_id=chat_id,
                        speaker="tool",
                        kind="tool",
                        ts=ts,
                        text=raw,
                        scope=(
                            f"chat:{chat_id}" if chat_id != "unknown" else "shared"
                        ),
                    )
            continue

        utterance = strip_meta(body)
        if not utterance:
            continue

        kind = "message"
        if speaker == "system":
            kind = "system"
        elif "[Voice client" in body:
            kind = "voice"

        yield ParsedEpisode(
            chat_id=chat_id,
            speaker=speaker,
            kind=kind,
            ts=ts,
            text=utterance,
            scope=f"chat:{chat_id}" if chat_id != "unknown" else "shared",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(app_config.DATA_DIR))
    parser.add_argument("--file", default=None, help="путь к message_history.md")
    parser.add_argument("--db", default=None, help="путь к memory.db")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tools", action="store_true", help="включить tool-результаты")
    parser.add_argument(
        "--no-sent",
        action="store_true",
        help="не восстанавливать реплики агента из max_send_message",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    source = Path(args.file) if args.file else data_dir / "message_history.md"
    if not source.exists():
        print(f"нет файла: {source}")
        return 1

    episodes = list(
        parse_history(
            source.read_text(encoding="utf-8"),
            include_tools=args.tools,
            include_sent=not args.no_sent,
        )
    )
    if args.limit:
        episodes = episodes[: args.limit]

    unique = {episode.key for episode in episodes}
    by_speaker = Counter(ep.speaker for ep in episodes)
    by_kind = Counter(ep.kind for ep in episodes)
    by_chat = Counter(ep.chat_id for ep in episodes)
    total_chars = sum(len(ep.text) for ep in episodes)

    print(f"блоков разобрано: {len(episodes)}, уникальных: {len(unique)}")
    print(f"  спикеры: {dict(by_speaker)}")
    print(f"  виды:    {dict(by_kind)}")
    print(f"  чаты:    {dict(by_chat.most_common(6))}")
    print(f"  символов: {total_chars}, время: {_span(episodes)}")

    db_path = Path(args.db) if args.db else resolve_db_path(data_dir)
    if not args.apply:
        print(f"\nпробный прогон; применить: --apply (БД: {db_path})")
        return 0

    store = MemoryStore(db_path)
    inserted = skipped = 0
    try:
        for index, ep in enumerate(episodes, start=1):
            row_id = store.add_episode(
                chat_id=ep.chat_id,
                speaker=ep.speaker,
                kind=ep.kind,
                ts=ep.ts,
                text=ep.text,
                ref=ep.ref,
                scope=ep.scope,
                commit=False,
            )
            if row_id is None:
                skipped += 1
            else:
                inserted += 1
            if index % COMMIT_EVERY == 0:
                store.connection.commit()
                print(f"  … {index}/{len(episodes)} (новых {inserted})")
        store.connection.commit()
    finally:
        store.close()

    print(f"\nготово: вставлено {inserted}, уже было {skipped}")
    print(f"БД: {db_path}")
    print("проверка: .venv/bin/python -c \"from server.memory_store import "
          f"MemoryStore; print(MemoryStore('{db_path}').stats())\"")
    return 0


def _span(episodes: list[ParsedEpisode]) -> str:
    if not episodes:
        return "—"
    first = datetime.fromtimestamp(min(ep.ts for ep in episodes))
    last = datetime.fromtimestamp(max(ep.ts for ep in episodes))
    return f"{first:%d.%m.%Y} → {last:%d.%m.%Y}"


if __name__ == "__main__":
    raise SystemExit(main())

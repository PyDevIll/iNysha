"""Split the bulky persistent profile into referenced files.

``persistent_memory.json`` is re-sent in every prompt, so stable prose blocks
(lessons, ideas, lore, server inventory) cost ~3.5 k tokens per turn just by
being there.  They are not lost: each block is written verbatim to a Markdown
file under ``DATA_DIR`` and the JSON keeps a pointer line the agent can follow
with ``fs_read`` when the topic actually comes up.

Dry run by default — prints the before/after token cost and the new JSON.
``--apply`` writes files, backs the original up and rewrites the JSON.

    .venv/bin/python scripts/compact_persistent_memory.py
    .venv/bin/python scripts/compact_persistent_memory.py --apply
    .venv/bin/python scripts/compact_persistent_memory.py --data-dir /tmp/copy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import config as app_config
from server.memory_text import preview

PERSISTENT_FILE = "persistent_memory.json"

# key -> (file name, heading, pointer text)
# Order matters: the pointer line replaces the block in the same position, so
# the rendered profile stays deterministic.
SPLIT_PLAN: dict[str, tuple[str, str, str]] = {
    "ideas_from_andrey": (
        "ideas.md",
        "Идеи Андрея",
        "data/ideas.md — идеи Андрея с рисками и вердиктами. Читать fs_read, "
        "когда обсуждаются его проекты.",
    ),
    "lessons_learned": (
        "lessons.md",
        "Уроки",
        "data/lessons.md — выученные уроки (окружение, сервисы, TTS, MAX, ключи). "
        "Читать fs_read, когда вопрос про инфраструктуру или прошлые грабли.",
    ),
    "lore_parad_zvezd": (
        "lore_parad_zvezd.md",
        "Парад звёзд — лор",
        "data/lore_parad_zvezd.md — только для чата «Парад звёзд» (Денис). "
        "Читать перед ответом ему.",
    ),
    "virtual_machines": (
        "servers.md",
        "Серверы",
        "data/servers.md и data/inysha_service.md §11 — self / pydevill / "
        "prolight: железо, диски, доступы (без секретов).",
    ),
}


def render_markdown(title: str, value) -> str:
    """Render one profile block as Markdown without losing a single field."""
    lines = [f"# {title}", ""]
    lines.append(
        f"_Извлечено из `{PERSISTENT_FILE}` "
        f"{datetime.now().strftime('%d.%m.%Y %H:%M')}._"
    )
    lines.append("")

    if isinstance(value, dict):
        for key, item in value.items():
            lines.extend(_render_value(str(key), item))
    elif isinstance(value, list):
        for index, item in enumerate(value, start=1):
            lines.extend(_render_item(index, item))
    else:
        lines.append(str(value))
    return "\n".join(lines).rstrip() + "\n"


def _render_item(index: int, item) -> list[str]:
    if not isinstance(item, dict):
        return [f"{index}. {item}", ""]
    title = next(
        (
            item[key]
            for key in ("title", "topic", "id", "name")
            if key in item and not isinstance(item[key], (dict, list))
        ),
        f"запись {index}",
    )
    lines = [f"## {index}. {title}", ""]
    for key, value in item.items():
        if value in ("", None) or str(value) == str(title):
            continue
        lines.extend(_render_value(str(key), value))
    lines.append("")
    return lines


def _render_value(key: str, value) -> list[str]:
    if isinstance(value, dict):
        out = [f"### {key}", ""]
        for sub_key, sub_value in value.items():
            out.extend(_render_value(str(sub_key), sub_value))
        return out
    if isinstance(value, list):
        out = [f"**{key}:**", ""]
        for item in value:
            out.append(f"- {item}")
        out.append("")
        return out
    return [f"- **{key}**: {value}"]


def compact(data: dict) -> tuple[dict, dict[str, str]]:
    """Return (new profile, markdown files to write)."""
    new_data: dict = {}
    files: dict[str, str] = {}
    for key, value in data.items():
        plan = SPLIT_PLAN.get(key)
        if plan is None or not isinstance(value, (dict, list)):
            # Already a pointer (string) — nothing left to split.
            new_data[key] = value
            continue
        file_name, heading, pointer = plan
        files[file_name] = render_markdown(heading, value)
        # Keep the original key order so the rendered profile is stable.
        new_data[key] = pointer
    return new_data, files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(app_config.DATA_DIR))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="перезаписать уже существующие .md (по умолчанию не трогаем)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    path = data_dir / PERSISTENT_FILE
    if not path.exists():
        print(f"нет файла: {path}")
        return 1

    from context_manager import render_persistent  # noqa: PLC0415 — после sys.path
    from context_manager import _token_count

    original_text = path.read_text(encoding="utf-8")
    original = json.loads(original_text)
    before = _token_count(render_persistent(original, max_tokens=10**9))
    new_data, files = compact(original)
    after = _token_count(render_persistent(new_data, max_tokens=10**9))

    print(f"профиль: {before} → {after} токенов на каждый ход "
          f"(−{before - after})")
    print(f"\nновый {PERSISTENT_FILE}:")
    print(json.dumps(new_data, ensure_ascii=False, indent=2))

    if new_data == original and not files:
        print("\nуже сжато, менять нечего")
        return 0

    existing = [name for name in files if (data_dir / name).exists()]
    if existing and not args.force:
        print(f"\nуже существуют, не перезаписываю: {', '.join(existing)}")
        print("перезаписать: --force")
        return 2

    if not args.apply:
        for name, text in files.items():
            print(f"\n--- {name}: {len(text)} символов, начало:")
            print("\n".join(text.splitlines()[:6]))
        print("\nэто пробный прогон; применить: --apply")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Optimistic concurrency: the service is live and the agent edits this file
    # itself. Writing a version built from a stale read would silently drop a
    # fact learned in between, so re-check the bytes before touching anything.
    if path.read_text(encoding="utf-8") != original_text:
        print(
            "файл изменился во время работы (агент пишет в него сам) — "
            "ничего не записано, запусти ещё раз"
        )
        return 3
    backup = path.with_suffix(f".json.bak_{stamp}")
    shutil.copy2(path, backup)
    for name, text in files.items():
        (data_dir / name).write_text(text, encoding="utf-8")
    path.write_text(
        json.dumps(new_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nготово. бэкап: {backup.name}")
    for name in files:
        print(f"  {name}: {preview((data_dir / name).read_text(encoding='utf-8'), 60)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

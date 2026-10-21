#!/usr/bin/env python3
"""Диагностика роста data/emergency_save.json.

Скрипт работает ТОЛЬКО с временной копией data/: оригинальные файлы
открываются на чтение, их размер/mtime/хеш печатаются до и после, чтобы
подтвердить, что они не изменены.

Что печатает:
  * сколько записей лежит в самом emergency_save.json;
  * сколько записей оказалось в ContextPool._all_entries после инициализации
    (restore из emergency_save.json + seed из last_compression.txt);
  * сколько записей и байт попадает в файл после одного вызова
    _emergency_save().

Запуск: .venv/bin/python scripts/diagnose_memory.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import context_manager  # noqa: E402
from context_manager import ContextPool  # noqa: E402


def _stat(path: Path) -> str:
    """Размер, mtime и sha256 файла одной строкой (или «нет файла»)."""
    if not path.exists():
        return "нет файла"
    data = path.read_bytes()
    return (
        f"size={len(data)} "
        f"mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime))} "
        f"sha256={hashlib.sha256(data).hexdigest()[:16]}"
    )


def _short(text: str, limit: int = 120) -> str:
    text = (text or "").replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def main() -> int:
    source = ROOT / "data"
    orig_save = source / "emergency_save.json"
    orig_last = source / "last_compression.txt"

    print("=== ОРИГИНАЛ (только чтение) ===")
    print(f"data/emergency_save.json: {_stat(orig_save)}")
    print(f"data/last_compression.txt: {_stat(orig_last)}")

    if not source.is_dir():
        print("data/ не найден — диагностика невозможна")
        return 1

    with tempfile.TemporaryDirectory(prefix="inysha-diag-") as tmp:
        copy_dir = Path(tmp) / "data"
        shutil.copytree(source, copy_dir)
        print(f"\n=== КОПИЯ data/ ===\n{copy_dir}")

        copy_save = copy_dir / "emergency_save.json"
        if copy_save.exists():
            raw = json.loads(copy_save.read_text(encoding="utf-8"))
            entries = raw.get("all_entries", raw if isinstance(raw, list) else [])
            print(
                f"входной emergency_save.json: all_entries={len(entries)}, "
                f"compressed_dicts={len(raw.get('compressed_dicts', []))}"
            )
            if entries:
                print(
                    f"первая запись: role={entries[0].get('role')} "
                    f"content={_short(entries[0].get('content') or '')!r}"
                )
                print(
                    f"последняя запись: role={entries[-1].get('role')} "
                    f"content={_short(entries[-1].get('content') or '')!r}"
                )
        else:
            print("входной emergency_save.json отсутствует")

        last_file = copy_dir / "last_compression.txt"
        print(
            "last_compression.txt: "
            + (f"{last_file.stat().st_size} байт" if last_file.exists() else "нет файла")
        )

        pool = ContextPool(data_dir=copy_dir)

        roles = Counter(e.role for e in pool._all_entries)
        print("\n=== ПОСЛЕ ИНИЦИАЛИЗАЦИИ ContextPool ===")
        print(f"_all_entries: {len(pool._all_entries)}")
        print(f"роли: {dict(roles)}")
        print(f"_compressed_dicts: {len(pool._compressed_dicts)}")
        if pool._compressed_dicts:
            stamps = [d.get("timestamp", "") for d in pool._compressed_dicts]
            print(f"timestamps сжатий: {stamps}")
        if pool._all_entries:
            first = pool._all_entries[0]
            print(
                f"первая запись в памяти: role={first.role} "
                f"content={_short(first.content)!r}"
            )

        if copy_save.exists():
            print(
                "источник _all_entries: emergency_save.json "
                f"({len(pool._all_entries)} записей после фильтрации/ремонта)"
            )
        if pool._compressed_dicts:
            print(
                "источник _compressed_dicts: emergency_save.json и/или "
                "last_compression.txt"
            )

        print("\n=== ОДИН ВЫЗОВ _emergency_save() ===")
        pool._emergency_save()
        written = json.loads(copy_save.read_text(encoding="utf-8"))
        written_entries = written.get("all_entries", [])
        print(f"записано записей: {len(written_entries)}")
        print(f"размер файла: {copy_save.stat().st_size} байт")
        print(f"compressed_dicts в файле: {len(written.get('compressed_dicts', []))}")
        if written_entries:
            print(
                f"первая запись в файле: role={written_entries[0].get('role')} "
                f"content={_short(written_entries[0].get('content') or '')!r}"
            )

        print("\n=== КОНСТАНТЫ ===")
        print(
            "EMERGENCY_KEEP_ENTRIES = "
            f"{getattr(context_manager, 'EMERGENCY_KEEP_ENTRIES', 'НЕТ (фикс не применён)')}"
        )
        print(
            "COMPRESS_KEEP_ENTRIES = "
            f"{getattr(context_manager, 'COMPRESS_KEEP_ENTRIES', 'НЕТ (фикс не применён)')}"
        )

    print("\n=== ОРИГИНАЛ ПОСЛЕ ДИАГНОСТИКИ (должен быть тем же) ===")
    print(f"data/emergency_save.json: {_stat(orig_save)}")
    print(f"data/last_compression.txt: {_stat(orig_last)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

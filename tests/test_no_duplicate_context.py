"""Проверка, что сборка контекста не дублирует сжатый блок памяти.

Сценарий: в data_dir лежит last_compression.txt с уникальным маркером.
ContextPool должен загрузить его ровно один раз (в _compressed_dicts),
не добавлять system/base-запись в _all_entries и показать маркер ровно
один раз в build_context().

Тесты работают только с временными каталогами и не трогают data/ проекта.
Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_manager import ContextPool


MARKER = "MARKER_X"

SUMMARY_TEXT = f"""## Key Facts
- Synthetic fact for the duplicate-context test: {MARKER}.

## Tool Results
- Synthetic tool result.

## Decisions
- Synthetic decision.

## Narrative
Synthetic narrative for the duplicate-context test.
"""


class NoDuplicateContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        (self.data_dir / "last_compression.txt").write_text(
            SUMMARY_TEXT, encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_marker_appears_exactly_once_in_build_context(self) -> None:
        pool = ContextPool(data_dir=self.data_dir)

        text, _ = pool.build_context("system prompt")

        self.assertEqual(text.count(MARKER), 1)

    def test_no_system_entries_and_no_duplicate_summary(self) -> None:
        pool = ContextPool(data_dir=self.data_dir)

        self.assertEqual(
            [entry.role for entry in pool._all_entries if entry.role == "system"],
            [],
        )
        self.assertEqual(
            len(pool._compressed_dicts),
            1,
            "last_compression.txt должен давать ровно один compressed-блок",
        )
        # В _all_entries сжатого текста быть не должно — только в _compressed_dicts.
        self.assertNotIn(
            MARKER,
            "\n".join(entry.content or "" for entry in pool._all_entries),
        )

        text, _ = pool.build_context("system prompt")
        self.assertEqual(text.count("PREVIOUS MEMORY SUMMARY"), 0)
        self.assertEqual(text.count("**Key Facts:**"), 1)


if __name__ == "__main__":
    unittest.main()

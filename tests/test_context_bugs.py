"""Регрессионные тесты для сборки контекста.

Проверяем три исправленных бага:
  1. last_compression.txt загружается ровно один раз (_compressed_dicts),
     а не ещё и как system/base-промпт в _all_entries;
  2. narrative из сжатого блока доходит до LLM (слой 3 в Agent);
  3. старые system/base-записи в emergency_save.json фильтруются при restore.

Тесты работают только с временными каталогами и не трогают data/ проекта.
Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent as agent_module
import context_manager
from context_manager import ContextPool


SUMMARY_TEXT = """## Key Facts
- Synthetic fact for the regression test.

## Tool Results
- Synthetic tool result.

## Decisions
- Synthetic decision.

## Narrative
Synthetic narrative for the regression test.
"""


class ContextBugsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        (self.data_dir / "last_compression.txt").write_text(
            SUMMARY_TEXT, encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_pool(self) -> ContextPool:
        return ContextPool(data_dir=self.data_dir)

    def test_last_compression_loaded_once_into_compressed(self) -> None:
        """(a) В _compressed_dicts ровно один блок из last_compression.txt."""
        pool = self._make_pool()

        self.assertEqual(len(pool._compressed_dicts), 1)
        self.assertEqual(
            pool._compressed_dicts[0]["narrative"],
            "Synthetic narrative for the regression test.",
        )

    def test_last_compression_not_stored_as_system_entry(self) -> None:
        """(b) В _all_entries нет system-записи с PREVIOUS MEMORY SUMMARY."""
        pool = self._make_pool()

        self.assertFalse(
            [
                entry
                for entry in pool._all_entries
                if entry.role == "system"
                or "PREVIOUS MEMORY SUMMARY" in (entry.content or "")
            ]
        )

    def test_build_context_contains_narrative(self) -> None:
        """(c) build_context() содержит narrative из входного блока."""
        pool = self._make_pool()

        text, _ = pool.build_context("system prompt")

        self.assertIn("**Narrative:**", text)
        self.assertIn("Synthetic narrative for the regression test.", text)

    def test_agent_layer3_contains_narrative(self) -> None:
        """narrative доходит до LLM через слой 3 (Agent._build_messages_for_llm)."""
        with mock.patch.object(context_manager, "DATA_DIR", self.data_dir):
            agent = agent_module.Agent(
                name="TEST",
                system_prompt="test system prompt",
                use_tools=False,
                save_history=False,
                api_key="test-key",
                base_url="http://127.0.0.1:1/v1",
            )

        messages = agent._build_messages_for_llm()
        all_content = "\n".join(
            message.get("content") or "" for message in messages
        )

        self.assertIn("**Narrative:**", all_content)
        self.assertIn("Synthetic narrative for the regression test.", all_content)

    def test_assign_messages_filters_system_base_prompt(self) -> None:
        """assign_messages не кладёт base-промпт в _all_entries."""
        pool = self._make_pool()

        pool.assign_messages(
            [
                {
                    "role": "system",
                    "name": "Deliria",
                    "content": "# **PREVIOUS MEMORY SUMMARY**\nold text",
                },
                {"role": "user", "content": "hello"},
            ]
        )

        self.assertEqual([entry.role for entry in pool._all_entries], ["user"])
        self.assertEqual(len(pool._compressed_dicts), 1)

    def test_legacy_emergency_save_system_entry_filtered(self) -> None:
        """Старый system-промпт в emergency_save.json отфильтровывается."""
        legacy_dir = self.data_dir / "legacy"
        legacy_dir.mkdir()
        legacy_save = {
            "all_entries": [
                {
                    "role": "system",
                    "name": "Deliria",
                    # _dict_to_entry добавляет [Time: ...] при восстановлении,
                    # поэтому проверяем вхождение текста, а не равенство.
                    "content": "# **PREVIOUS MEMORY SUMMARY**\nold text",
                },
                {"role": "user", "content": "hello"},
            ],
            "compressed_dicts": [],
        }
        (legacy_dir / "emergency_save.json").write_text(
            json.dumps(legacy_save, ensure_ascii=False), encoding="utf-8"
        )

        pool = ContextPool(data_dir=legacy_dir)

        self.assertEqual([entry.role for entry in pool._all_entries], ["user"])
        self.assertFalse(
            [
                entry
                for entry in pool._all_entries
                if "PREVIOUS MEMORY SUMMARY" in (entry.content or "")
            ]
        )

        # Следующее сохранение не должно вернуть system-запись обратно.
        pool._emergency_save()
        written = json.loads(
            (legacy_dir / "emergency_save.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [entry["role"] for entry in written["all_entries"]], ["user"]
        )


if __name__ == "__main__":
    unittest.main()

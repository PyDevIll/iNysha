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

import asyncio
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
from context_manager import (
    COMPRESS_KEEP_ENTRIES,
    EMERGENCY_KEEP_ENTRIES,
    ContextPool,
    MemoryEntry,
)


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

    # ── tool-call / tool-result pairing ───────────────────────────
    @staticmethod
    def _dangling_tool_call() -> dict:
        """An assistant tool call whose result was never written (crash)."""
        return {
            "role": "assistant",
            "content": "",
            "time": "21.10.2026, 06:50",
            "tool_calls": [
                {
                    "id": "call_interrupted",
                    "type": "function",
                    "function": {"name": "exec_python", "arguments": "{}"},
                }
            ],
        }

    def test_repair_tool_sequence_answers_dangling_call(self) -> None:
        repaired = context_manager.repair_tool_sequence(
            [
                {"role": "user", "content": "start"},
                self._dangling_tool_call(),
                {"role": "user", "content": "next"},
            ]
        )

        self.assertEqual(
            [m["role"] for m in repaired], ["user", "assistant", "tool", "user"]
        )
        self.assertEqual(repaired[2]["tool_call_id"], "call_interrupted")
        self.assertEqual(repaired[2]["name"], "exec_python")
        agent_module.validate_message_sequence(repaired)

    def test_repair_tool_sequence_converts_orphan_tool(self) -> None:
        repaired = context_manager.repair_tool_sequence(
            [
                {
                    "role": "tool",
                    "tool_call_id": "missing_call",
                    "name": "exec_python",
                    "content": "orphan output",
                },
                {"role": "user", "content": "hi"},
            ]
        )

        self.assertEqual([m["role"] for m in repaired], ["user", "user"])
        self.assertIn("orphan output", repaired[0]["content"])

    def test_validate_message_sequence_rejects_unanswered_call(self) -> None:
        with self.assertRaises(ValueError):
            agent_module.validate_message_sequence([self._dangling_tool_call()])

    def test_restore_repairs_crash_interrupted_tool_call(self) -> None:
        save = {
            "all_entries": [
                self._dangling_tool_call(),
                {"role": "user", "content": "next", "time": "21.10.2026, 07:14"},
            ],
            "compressed_dicts": [],
        }
        (self.data_dir / "emergency_save.json").write_text(
            json.dumps(save, ensure_ascii=False), encoding="utf-8"
        )

        pool = self._make_pool()

        self.assertEqual(
            [entry.role for entry in pool._all_entries],
            ["assistant", "tool", "user"],
        )
        agent_module.validate_message_sequence(
            [pool._entry_to_openai_dict(entry) for entry in pool._all_entries]
        )

    def test_agent_build_repairs_dangling_tool_call(self) -> None:
        with mock.patch.object(context_manager, "DATA_DIR", self.data_dir):
            agent = agent_module.Agent(
                name="TEST",
                system_prompt="test system prompt",
                use_tools=False,
                save_history=False,
                api_key="test-key",
                base_url="http://127.0.0.1:1/v1",
            )

        agent.messages.append(self._dangling_tool_call())
        agent.messages.append({"role": "user", "content": "next"})

        messages = agent._build_messages_for_llm()

        agent_module.validate_message_sequence(messages)
        self.assertTrue(
            any(
                m.get("role") == "tool"
                and m.get("tool_call_id") == "call_interrupted"
                for m in messages
            )
        )

    def test_time_prefix_not_stacked_on_round_trip(self) -> None:
        pool = self._make_pool()
        pool.append(
            {"role": "user", "content": "hello", "time": "21.10.2026, 07:00"}
        )

        entry = pool._all_entries[0]
        once = pool._dict_to_entry(pool._entry_to_dict(entry))
        twice = pool._dict_to_entry(pool._entry_to_dict(once))

        self.assertEqual(twice.content.count("[Time:"), 1)

    # ── рост emergency_save.json / обрезка после компрессии ───────
    def test_emergency_save_keeps_only_last_40_entries(self) -> None:
        """Из 500 записей в файл попадают ровно последние EMERGENCY_KEEP_ENTRIES."""
        pool = self._make_pool()
        pool._all_entries = [
            MemoryEntry(
                role="user",
                content=f"entry-{i}",
                time="21.10.2026, 08:00",
            )
            for i in range(500)
        ]

        pool._emergency_save()

        written = json.loads(
            (self.data_dir / "emergency_save.json").read_text(encoding="utf-8")
        )
        entries = written["all_entries"]

        self.assertEqual(EMERGENCY_KEEP_ENTRIES, 40)
        self.assertEqual(len(entries), EMERGENCY_KEEP_ENTRIES)
        self.assertEqual(
            [entry["content"] for entry in entries],
            [f"entry-{i}" for i in range(500 - EMERGENCY_KEEP_ENTRIES, 500)],
        )

    def test_compress_keeps_at_most_five_entries(self) -> None:
        """После compress() в _all_entries остаются только 5 последних записей."""
        pool = self._make_pool()
        entries = [
            MemoryEntry(role="user", content=f"old-{i}", time="21.10.2026, 08:00")
            for i in range(50)
        ]
        # Позиция 45 — tool-результат без assistant tool_call. Обрезка до
        # последних 5 записей начнётся именно с него: «сироту» нужно отбросить.
        entries[45] = MemoryEntry(
            role="tool",
            content="orphan tool output",
            time="21.10.2026, 08:00",
            tool_name="exec_python",
            tool_call_id="call_orphan",
        )
        pool._all_entries = entries

        asyncio.run(pool.compress(_FakeHelper(SUMMARY_TEXT), force=True))

        self.assertEqual(COMPRESS_KEEP_ENTRIES, 5)
        self.assertLessEqual(len(pool._all_entries), COMPRESS_KEEP_ENTRIES)
        self.assertNotEqual(pool._all_entries[0].role, "tool")
        agent_module.validate_message_sequence(
            [pool._entry_to_openai_dict(entry) for entry in pool._all_entries]
        )

        # Снимок на диск тоже не должен вернуть старые записи.
        pool._emergency_save()
        written = json.loads(
            (self.data_dir / "emergency_save.json").read_text(encoding="utf-8")
        )
        self.assertLessEqual(len(written["all_entries"]), COMPRESS_KEEP_ENTRIES)

    def test_restore_from_trimmed_save_is_valid(self) -> None:
        """Восстановление из обрезанного файла не падает и валидно по tool-call."""
        save = {
            "all_entries": [
                {
                    "role": "tool",
                    "tool_call_id": "call_orphan",
                    "name": "exec_python",
                    "content": "orphan output after trim",
                },
                {"role": "user", "content": "after trim"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_kept",
                            "type": "function",
                            "function": {"name": "exec_python", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_kept",
                    "name": "exec_python",
                    "content": "ok",
                },
                {"role": "user", "content": "done"},
            ],
            "compressed_dicts": [],
        }
        (self.data_dir / "emergency_save.json").write_text(
            json.dumps(save, ensure_ascii=False), encoding="utf-8"
        )

        pool = self._make_pool()

        self.assertTrue(pool._all_entries)
        self.assertNotEqual(pool._all_entries[0].role, "tool")
        agent_module.validate_message_sequence(
            [pool._entry_to_openai_dict(entry) for entry in pool._all_entries]
        )


class _FakeHelperMessages:
    """Заглушка messages для helper_agent: assign_messages ничего не делает."""

    def assign_messages(self, messages: list[dict]) -> None:
        self.last = messages


class _FakeHelper:
    """Заглушка helper_agent: сразу возвращает заданный текст сжатия."""

    def __init__(self, text: str):
        self._text = text
        self.messages = _FakeHelperMessages()

    async def llm_request(self, enable_reasoning: bool = False) -> dict:
        return {"choices": [{"message": {"content": self._text}}]}


if __name__ == "__main__":
    unittest.main()

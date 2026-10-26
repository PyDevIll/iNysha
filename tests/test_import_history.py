"""Импорт истории: разбор транскрипта, идемпотентность, отсутствие потерь.

message_history.md — единственный транскрипт агента. Импорт не должен ни
дублировать строки при повторном запуске, ни оставлять в тексте служебные
маркеры MAX, ни писать в боевую БД при прогоне на копии.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "import_history.py"
_spec = importlib.util.spec_from_file_location("import_history", SCRIPT)
import_history = importlib.util.module_from_spec(_spec)
# dataclass() looks the module up in sys.modules while decorating.
sys.modules["import_history"] = import_history
_spec.loader.exec_module(import_history)  # type: ignore[union-attr]

from server.memory_store import MemoryStore, close_store

HISTORY = """
**user:** [18.10.2026, 14:39]
[Time: 18.10.2026, 14:39][MAX messenger | chat_id=326963375 | user_id=1 | sender=Андрей | trusted=true]
Пуп
---

**assistant:** [18.10.2026, 14:39]
Привет, Андрей.
---

**assistant:** [18.10.2026, 14:40]
---

**tool:** [18.10.2026, 14:40]
[Time: 18.10.2026, 14:40]{'ok': True, 'result': {'body': {'text': 'доставлено'}}}
---

**user:** [18.10.2026, 14:41]
[Time: 18.10.2026, 14:41][MAX messenger | chat_id=367837204 | user_id=2 | sender=Денис]
Лор
---

**user:** [18.10.2026, 14:42]
[Time: 18.10.2026, 14:42][Voice client: web] голосом
---
"""

SENT_TOOL = """
**tool:** [18.10.2026, 15:00]
[Time: 18.10.2026, 15:00]{'ok': True, 'result': {'recipient': {'chat_id': 326963375}, 'body': {'mid': 'mid.123', 'text': 'Работаем!'}, 'sender': {'is_bot': True, 'first_name': 'iНюша'}}}
---

**tool:** [18.10.2026, 15:01]
[Time: 18.10.2026, 15:01]{'ok': True, 'result': {'recipient': {'chat_id': 326963375}, 'body': {'mid': 'mid.124', 'text': 'от другого'}, 'sender': {'is_bot': False}}}
---

**tool:** [18.10.2026, 15:02]
[Time: 18.10.2026, 15:02]{'ok': False, 'error': 'timeout'}
---
"""


class ParseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = list(import_history.parse_history(HISTORY))

    def test_tools_and_empty_assistants_are_skipped(self) -> None:
        speakers = [ep.speaker for ep in self.episodes]
        self.assertEqual(speakers, ["human", "agent", "human", "human"])

    def test_assistant_inherits_chat_from_previous_user(self) -> None:
        self.assertEqual(self.episodes[1].chat_id, "326963375")

    def test_chat_switch_is_followed(self) -> None:
        self.assertEqual(self.episodes[2].chat_id, "367837204")

    def test_markers_are_stripped_from_text(self) -> None:
        self.assertEqual(self.episodes[0].text, "Пуп")
        self.assertNotIn("[Time:", self.episodes[0].text)

    def test_voice_is_detected(self) -> None:
        self.assertEqual(self.episodes[3].kind, "voice")

    def test_scope_follows_chat(self) -> None:
        self.assertEqual(self.episodes[0].scope, "chat:326963375")

    def test_tools_can_be_included(self) -> None:
        with_tools = list(import_history.parse_history(HISTORY, include_tools=True))
        self.assertEqual(len(with_tools), 5)


class SentMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = list(import_history.parse_history(SENT_TOOL))

    def test_bot_send_becomes_agent_episode(self) -> None:
        self.assertEqual(len(self.episodes), 1, self.episodes)
        episode = self.episodes[0]
        self.assertEqual(episode.speaker, "agent")
        self.assertEqual(episode.text, "Работаем!")
        self.assertEqual(episode.ref, "mid.123")
        self.assertEqual(episode.chat_id, "326963375")

    def test_foreign_message_is_not_attributed_to_agent(self) -> None:
        texts = [ep.text for ep in self.episodes]
        self.assertNotIn("от другого", texts)

    def test_failed_tool_call_is_ignored(self) -> None:
        self.assertNotIn("timeout", " ".join(ep.text for ep in self.episodes))

    def test_truncated_payload_does_not_crash(self) -> None:
        broken = (
            "**tool:** [18.10.2026, 15:00]\n"
            "[Time: 18.10.2026, 15:00]{'ok': True, 'result': {'body': {'text': 'обре"
        )
        self.assertEqual(list(import_history.parse_history(broken)), [])

    def test_sent_can_be_disabled(self) -> None:
        self.assertEqual(
            list(import_history.parse_history(SENT_TOOL, include_sent=False)), []
        )


class ImportTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        (self.data_dir / "message_history.md").write_text(
            HISTORY.strip() + "\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        close_store()
        self._tmp.cleanup()

    def run_import(self, *args: str) -> int:
        return import_history.main(["--data-dir", str(self.data_dir), *args])

    def store(self) -> MemoryStore:
        return MemoryStore(self.data_dir / "memory.db")

    def test_dry_run_writes_nothing(self) -> None:
        self.assertEqual(self.run_import(), 0)
        self.assertFalse((self.data_dir / "memory.db").exists())

    def test_apply_inserts_rows(self) -> None:
        self.assertEqual(self.run_import("--apply"), 0)
        store = self.store()
        try:
            self.assertEqual(store.stats()["episodes"], 4)
            row = store.connection.execute(
                "SELECT text, kind FROM episodes WHERE speaker='human' "
                "ORDER BY ts LIMIT 1").fetchone()
        finally:
            store.close()
        self.assertEqual(row["text"], "Пуп")
        self.assertEqual(row["kind"], "message")

    def test_second_import_is_idempotent(self) -> None:
        self.run_import("--apply")
        store = self.store()
        try:
            first = store.stats()["episodes"]
        finally:
            store.close()
        self.run_import("--apply")
        store = self.store()
        try:
            second = store.stats()["episodes"]
        finally:
            store.close()
        self.assertEqual(first, second)

    def test_episodes_are_searchable_by_lemma(self) -> None:
        self.run_import("--apply")
        store = self.store()
        try:
            ids = store.connection.execute(
                "SELECT rowid FROM episodes_fts WHERE episodes_fts MATCH ?",
                ('"привет"',),
            ).fetchall()
        finally:
            store.close()
        self.assertEqual(len(ids), 1)

    def test_missing_file_is_reported(self) -> None:
        (self.data_dir / "message_history.md").unlink()
        self.assertEqual(self.run_import(), 1)


if __name__ == "__main__":
    unittest.main()

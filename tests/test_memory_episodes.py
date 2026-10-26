"""Эпизоды: запись из горячего пути, дедуп, изоляция от живой памяти.

Проверяем три вещи, которые легко сломать незаметно:
  * эпизод пишется из того же хука, что и message_history.md (один источник);
  * сбой SQLite не ломает ход (память — надстройка, а не точка отказа);
  * пул, созданный с временным data_dir, не пишет в боевой data/memory.db.

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

from context_manager import ContextPool, MemoryEntry
from server import config as app_config
from server.context import RequestContext, current_request
from server.memory_episodes import (
    EpisodeLog,
    extract_chat_id,
    parse_entry_time,
    resolve_db_path,
    strip_meta,
)
from server.memory_store import MemoryStore, close_store
from datetime import datetime


def entry(role: str = "user", content: str = "привет", **kw) -> MemoryEntry:
    return MemoryEntry(
        role=role,
        content=content,
        time=kw.pop("time", "26.10.2026, 06:37"),
        **kw,
    )


class ParsingTest(unittest.TestCase):
    def test_strip_meta_removes_transport_markers(self) -> None:
        raw = (
            "[Time: 26.10.2026, 06:37][MAX messenger | chat_id=326963375 "
            "| user_id=1 | sender=Андрей | trusted=true]\nПривет"
        )
        self.assertEqual(strip_meta(raw), "Привет")

    def test_strip_meta_keeps_real_brackets(self) -> None:
        self.assertEqual(strip_meta("[Итог] всё работает"), "[Итог] всё работает")

    def test_strip_meta_handles_voice_and_desktop(self) -> None:
        self.assertEqual(
            strip_meta("[Time: 26.10.2026, 06:37][Voice client: web] текст"),
            "текст",
        )

    def test_extract_chat_id(self) -> None:
        self.assertEqual(extract_chat_id("chat_id=-76940393420843 | x"), "-76940393420843")
        self.assertIsNone(extract_chat_id("нет маркера"))

    def test_parse_entry_time(self) -> None:
        ts = parse_entry_time("26.10.2026, 06:37")
        self.assertIsNotNone(ts)
        self.assertEqual(datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M"),
                         "26.10.2026 06:37")
        self.assertIsNone(parse_entry_time("кривая дата"))
        self.assertIsNone(parse_entry_time(""))


class PathTest(unittest.TestCase):
    def test_tmp_dir_gets_its_own_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_db_path(Path(tmp))
            self.assertEqual(resolved, Path(tmp).resolve() / "memory.db")
            self.assertNotEqual(
                resolved, Path(app_config.MEMORY_DB_PATH).resolve()
            )

    def test_live_data_dir_uses_configured_path(self) -> None:
        self.assertEqual(
            resolve_db_path(Path(app_config.DATA_DIR)),
            Path(app_config.MEMORY_DB_PATH),
        )


class EpisodeLogTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.log = EpisodeLog(self.data_dir)

    def tearDown(self) -> None:
        self.log.close()
        close_store()
        current_request.set(None)
        self._tmp.cleanup()

    def rows(self) -> list:
        return self.log.store.connection.execute(
            "SELECT * FROM episodes ORDER BY id"
        ).fetchall()

    def test_marks_user_space_comes_from_markers(self) -> None:
        self.log.record(entry(content=(
            "[Time: 26.10.2026, 06:37][MAX messenger | chat_id=326963375 | "
            "sender=Андрей]\nПривет"
        )))
        row = self.rows()[0]
        self.assertEqual(row["speaker"], "human")
        self.assertEqual(row["chat_id"], "326963375")
        self.assertEqual(row["text"], "Привет")
        self.assertEqual(row["scope"], "shared")

    def test_request_context_wins_over_markers(self) -> None:
        current_request.set(RequestContext(
            request_id="r1", source="max", chat_id="555"))
        self.log.record(entry(content="без маркеров"))
        row = self.rows()[0]
        self.assertEqual(row["chat_id"], "555")
        self.assertEqual(row["scope"], "chat:555")

    def test_dedupe_identical_append(self) -> None:
        self.assertEqual(self.log.record(entry()), 1)
        self.assertIsNone(self.log.record(entry()))
        self.assertEqual(len(self.rows()), 1)

    def test_same_text_new_minute_is_a_new_episode(self) -> None:
        self.log.record(entry(time="26.10.2026, 06:37"))
        self.log.record(entry(time="26.10.2026, 06:38"))
        self.assertEqual(len(self.rows()), 2)

    def test_empty_content_is_skipped(self) -> None:
        self.assertIsNone(self.log.record(entry(content="")))
        self.assertIsNone(self.log.record(entry(content="[Time: 1]")))
        self.assertEqual(self.rows(), [])

    def test_tool_results_are_off_by_default(self) -> None:
        self.assertIsNone(self.log.record(entry(role="tool", content="{'ok': 1}")))
        self.assertEqual(self.rows(), [])

    def test_tool_results_can_be_enabled(self) -> None:
        log = EpisodeLog(self.data_dir, store_tools=True)
        try:
            log.record(entry(role="tool", content="{'ok': 1}"))
            row = log.store.connection.execute(
                "SELECT speaker, kind FROM episodes").fetchone()
        finally:
            log.close()
        self.assertEqual((row["speaker"], row["kind"]), ("tool", "tool"))

    def test_voice_turn_is_marked(self) -> None:
        current_request.set(RequestContext(
            request_id="r1", source="voice", chat_id="1", voice_session_id="vs1"))
        self.log.record(entry(content="скажи вслух"))
        self.assertEqual(self.rows()[0]["kind"], "voice")

    def test_disabled_log_writes_nothing(self) -> None:
        log = EpisodeLog(self.data_dir / "off", enabled=False)
        self.assertIsNone(log.record(entry()))
        self.assertFalse((self.data_dir / "off" / "memory.db").exists())
        log.close()

    def test_lemmas_are_stored_for_search(self) -> None:
        self.log.record(entry(content="Сервер перезапустили дважды"))
        row = self.rows()[0]
        self.assertIn("сервер", row["lemma_text"])
        self.assertIn("перезапустить", row["lemma_text"])

    def test_db_failure_does_not_raise(self) -> None:
        class Broken:
            def add_episode(self, **kwargs):
                raise RuntimeError("disk is on fire")

        self.log._store = Broken()  # type: ignore[assignment]
        self.assertIsNone(self.log.record(entry(content="текст")))
        self.log._store = None


class ContextPoolIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        close_store()
        self._tmp.cleanup()

    def test_append_with_save_records_episode(self) -> None:
        pool = ContextPool(data_dir=self.data_dir)
        pool.append({"role": "user", "name": "User", "content": "привет"},
                    save=True)
        db = self.data_dir / "memory.db"
        self.assertTrue(db.exists(), "эпизод должен попасть в БД пула")
        store = MemoryStore(db)
        try:
            self.assertEqual(store.stats()["episodes"], 1)
        finally:
            store.close()

    def test_append_without_save_does_not_write(self) -> None:
        pool = ContextPool(data_dir=self.data_dir)
        pool.append({"role": "user", "name": "User", "content": "привет"})
        self.assertFalse((self.data_dir / "memory.db").exists())

    def test_tmp_pool_never_touches_live_db(self) -> None:
        pool = ContextPool(data_dir=self.data_dir)
        pool.append({"role": "user", "name": "User", "content": "x"}, save=True)
        self.assertNotEqual(
            pool._episodes.path, Path(app_config.MEMORY_DB_PATH).resolve()
        )


if __name__ == "__main__":
    unittest.main()

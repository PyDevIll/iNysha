"""Источники фактов: что реально прочитано, а что модель выдумала.

Проверяем то, ради чего таблица существует:
  * канонизация путей и URL (http/https, utm, якорь — один и тот же источник);
  * запись только после успешного результата (ошибка инструмента — не источник);
  * извлечение URL из поисковой выдачи с ограничением;
  * окно цитирования: старее 7 дней — цитировать нельзя;
  * хук в agent._execute_single_tool действительно пишет источник;
  * метка для промпта короткая и с датой.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import context_manager
import agent as agent_module
from server import read_sources
from server.memory_store import MemoryStore, close_store, now_ts


class CanonicalTest(unittest.TestCase):
    def test_url_is_normalised(self) -> None:
        self.assertEqual(
            read_sources.normalize_url(
                "http://Example.COM/page/?utm_source=x&b=2#frag"
            ),
            "https://example.com/page?b=2",
        )

    def test_http_and_https_are_the_same_source(self) -> None:
        self.assertEqual(
            read_sources.normalize_url("http://example.com/a"),
            read_sources.normalize_url("https://example.com/a"),
        )

    def test_default_port_is_dropped(self) -> None:
        self.assertEqual(
            read_sources.normalize_url("https://example.com:443/a"),
            "https://example.com/a",
        )

    def test_file_path_is_absolute(self) -> None:
        kind, ref = read_sources.canonical_ref("data/lessons.md")
        self.assertEqual(kind, "file")
        self.assertTrue(Path(ref).is_absolute())
        self.assertTrue(ref.endswith("data/lessons.md"))

    def test_tilde_is_expanded(self) -> None:
        kind, ref = read_sources.canonical_ref("~/note.md")
        self.assertEqual(kind, "file")
        self.assertNotIn("~", ref)

    def test_bare_word_is_not_a_source(self) -> None:
        self.assertIsNone(read_sources.canonical_ref("сервер"))
        self.assertIsNone(read_sources.canonical_ref(""))
        self.assertIsNone(read_sources.canonical_ref("ftp://x/y"))


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "memory.db")
        self.file = Path(self._tmp.name) / "note.md"
        self.file.write_text("hello", encoding="utf-8")

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        self._tmp.cleanup()


class RecordingTest(Base):
    def test_fs_read_records_size_and_mtime(self) -> None:
        count = read_sources.record_tool_read(
            self.store, "fs_read", {"file": str(self.file)}, "hello"
        )
        self.assertEqual(count, 1)
        row = self.store.get_source("file", str(self.file.resolve()))
        self.assertIsNotNone(row)
        self.assertEqual(row["size"], 5)
        self.assertIsNotNone(row["mtime"])

    def test_fs_read_docx_uses_the_same_argument(self) -> None:
        doc = Path(self._tmp.name) / "a.docx"
        doc.write_bytes(b"PK\x03\x04")
        read_sources.record_tool_read(
            self.store, "fs_read_docx", {"file": str(doc)}, "текст"
        )
        self.assertIsNotNone(self.store.get_source("file", str(doc.resolve())))

    def test_browse_records_the_url_argument(self) -> None:
        read_sources.record_tool_read(
            self.store, "tavily_browse",
            {"url": "https://example.com/docs?utm_medium=x"}, "text",
        )
        self.assertIsNotNone(
            self.store.get_source("web", "https://example.com/docs")
        )

    def test_error_result_is_not_a_source(self) -> None:
        for text in ("Error: not found", "File not found: x", "Not a .docx file"):
            with self.subTest(text=text):
                count = read_sources.record_tool_read(
                    self.store, "fs_read", {"file": str(self.file)}, text
                )
                self.assertEqual(count, 0)

    def test_search_records_urls_from_the_result(self) -> None:
        result = (
            "1. Заголовок https://example.com/a\n"
            "2. Другой https://example.com/b?utm_source=t\n"
            "3. Тот же https://example.com/a#top\n"
        )
        count = read_sources.record_tool_read(
            self.store, "tavily_search", {"query": "x"}, result
        )
        self.assertEqual(count, 2, "дубликат и utm должны схлопнуться")
        self.assertIsNotNone(self.store.get_source("web", "https://example.com/a"))

    def test_search_is_capped(self) -> None:
        result = "\n".join(f"https://example.com/{i}" for i in range(30))
        read_sources.record_tool_read(
            self.store, "tavily_search", {"query": "x"}, result
        )
        rows = self.store.connection.execute("SELECT count(*) c FROM sources")
        self.assertLessEqual(int(rows.fetchone()["c"]),
                             read_sources.MAX_SOURCES_PER_CALL)

    def test_untracked_tool_is_ignored(self) -> None:
        self.assertEqual(read_sources.record_tool_read(
            self.store, "exec_shell", {"command": "cat x"}, str(self.file)
        ), 0)
        rows = self.store.connection.execute("SELECT count(*) c FROM sources")
        self.assertEqual(int(rows.fetchone()["c"]), 0)

    def test_repeat_read_updates_instead_of_duplicating(self) -> None:
        for _ in range(3):
            read_sources.record_tool_read(
                self.store, "fs_read", {"file": str(self.file)}, "hello"
            )
        row = self.store.get_source("file", str(self.file.resolve()))
        self.assertEqual(row["reads"], 3)


class ResolveTest(Base):
    def record(self, raw: str = "") -> dict:
        read_sources.record_tool_read(
            self.store, "fs_read", {"file": str(self.file)}, "hello"
        )
        resolved = read_sources.resolve_source(self.store, raw or str(self.file))
        self.assertIsNotNone(resolved)
        return resolved

    def test_read_file_resolves(self) -> None:
        resolved = self.record()
        self.assertEqual(resolved["kind"], "file")
        self.assertIsNotNone(resolved["mtime"])

    def test_invented_url_does_not_resolve(self) -> None:
        self.assertIsNone(read_sources.resolve_source(
            self.store, "https://made-up.example/page"
        ))

    def test_unread_file_does_not_resolve(self) -> None:
        other = Path(self._tmp.name) / "other.md"
        other.write_text("x", encoding="utf-8")
        self.record()
        self.assertIsNone(read_sources.resolve_source(self.store, str(other)))

    def test_source_older_than_the_window_does_not_resolve(self) -> None:
        self.record()
        old = now_ts() - 8 * 86400
        self.store.connection.execute(
            "UPDATE sources SET last_seen=?", (old,)
        )
        self.store.connection.commit()
        self.assertIsNone(read_sources.resolve_source(self.store, str(self.file)))

    def test_label_is_short_and_dated(self) -> None:
        resolved = self.record()
        text = read_sources.label(resolved)
        self.assertTrue(text.startswith("[file "), text)
        self.assertIn("note.md", text)

    def test_web_label_has_host_and_age(self) -> None:
        read_sources.record_tool_read(
            self.store, "tavily_browse", {"url": "https://example.com/x"}, "ok"
        )
        resolved = read_sources.resolve_source(self.store, "https://example.com/x")
        self.assertIn("example.com", read_sources.label(resolved))
        self.assertIn("0 дн.", read_sources.label(resolved))


class AgentHookTest(unittest.TestCase):
    """Хук в агенте: источник пишется после настоящего выполнения инструмента."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.file = self.tmp / "note.md"
        self.file.write_text("содержимое", encoding="utf-8")
        self.store = MemoryStore(self.tmp / "memory.db")

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        self._tmp.cleanup()

    def test_fs_stat_result_creates_a_source(self) -> None:
        with mock.patch.object(context_manager, "DATA_DIR", self.tmp), \
             mock.patch.object(agent_module, "get_store", return_value=self.store):
            instance = agent_module.Agent(
                name="TEST",
                system_prompt="test",
                use_tools=True,
                save_history=False,
                api_key="test-key",
                base_url="http://127.0.0.1:1/v1",
            )
            call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="fs_stat",
                    arguments=json.dumps({"path": str(self.file)}),
                ),
            )
            message = asyncio.run(instance._execute_single_tool(call))

        self.assertEqual(message["role"], "tool")
        self.assertNotIn("Error", message["content"][:20])
        row = self.store.get_source("file", str(self.file.resolve()))
        self.assertIsNotNone(row, "хук не записал источник")

    def test_failed_tool_creates_no_source(self) -> None:
        with mock.patch.object(context_manager, "DATA_DIR", self.tmp), \
             mock.patch.object(agent_module, "get_store", return_value=self.store):
            instance = agent_module.Agent(
                name="TEST",
                system_prompt="test",
                use_tools=True,
                save_history=False,
                api_key="test-key",
                base_url="http://127.0.0.1:1/v1",
            )
            call = SimpleNamespace(
                id="call-2",
                function=SimpleNamespace(
                    name="fs_stat",
                    arguments=json.dumps({"path": str(self.tmp / "missing.md")}),
                ),
            )
            asyncio.run(instance._execute_single_tool(call))

        rows = self.store.connection.execute("SELECT count(*) c FROM sources")
        self.assertEqual(int(rows.fetchone()["c"]), 0)


if __name__ == "__main__":
    unittest.main()

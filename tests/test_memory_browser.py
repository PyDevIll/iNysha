"""Просмотрщик памяти: только чтение, без утечек и без падений на пустой базе.

Проверяем то, что легко сломать незаметно:
  * соединение действительно read-only (запись через него невозможна);
  * HTML экранируется — имя факта не может стать тегом;
  * фильтры и FTS работают как у агента (тот же построитель запроса);
  * страницы рисуются на пустой и на заполненной базе;
  * токен закрывает доступ, POST запрещён.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.memory_store import MemoryStore, close_store

SCRIPT = ROOT / "scripts" / "memory_browser.py"
_spec = importlib.util.spec_from_file_location("memory_browser", SCRIPT)
browser = importlib.util.module_from_spec(_spec)
sys.modules["memory_browser"] = browser
_spec.loader.exec_module(browser)  # type: ignore[union-attr]


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.upsert_node(
            kind="fact", name="лимит ElevenLabs", name_norm="лимит elevenlabs",
            node_text="10000 символов в месяц",
            data=json.dumps({
                "text": "10000 символов в месяц",
                "source": {"kind": "web", "ref": "https://example.com/limits",
                           "last_seen": 1_790_000_000},
            }, ensure_ascii=False),
            origin="web", confidence=0.6,
        )
        self.store.upsert_node(
            kind="rule", name="<script>alert(1)</script>",
            name_norm="<script>alert(1)</script>",
            node_text="опасное имя",
            data=json.dumps({"text": "опасное имя"}, ensure_ascii=False),
            origin="operator", confidence=0.9,
        )
        self.store.add_episode(
            chat_id="326963375", speaker="human", kind="message",
            ts=1_790_000_000, text="Про память и сервер prolight",
        )
        self.store.add_episode(
            chat_id="326963375", speaker="agent", kind="message",
            ts=1_790_000_060, text="Запомнила про prolight",
        )
        self.store.connection.execute(
            "INSERT INTO pending_facts(chat_id, payload, reason, status, created_at)"
            " VALUES('326963375', ?, 'extractor', 'pending', 1)",
            (json.dumps({"kind": "fact", "name": "кандидат",
                         "text": "из истории", "confidence": 0.8},
                        ensure_ascii=False),),
        )
        self.store.record_source(
            kind="file", ref=str(Path(self._tmp.name) / "note.md"),
            size=5, mtime=1,
        )
        self.store.record_source(
            kind="web", ref="https://example.com/docs", title="Docs",
        )
        self.store.connection.commit()
        self.conn = browser.connect_ro(self.db_path)

    def tearDown(self) -> None:
        self.conn.close()
        self.store.close()
        close_store()
        self._tmp.cleanup()


class ReadOnlyTest(Base):
    def test_write_through_the_viewer_is_impossible(self) -> None:
        with self.assertRaises(Exception):
            self.conn.execute("DELETE FROM nodes")
            self.conn.commit()

    def test_schema_is_not_migrated_by_opening(self) -> None:
        before = self.store.connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()["value"]
        self.conn.close()
        self.conn = browser.connect_ro(self.db_path)
        after = self.store.connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()["value"]
        self.assertEqual(before, after)


class PageTest(Base):
    def test_overview_counts(self) -> None:
        text = browser.page_overview(self.conn, self.db_path)
        self.assertIn("Обзор", text)
        self.assertIn("эпизоды", text)
        self.assertIn(">2<", text, "два эпизода в карточке/таблице")
        self.assertIn("схема v", text)

    def test_nodes_page_lists_facts_and_provenance(self) -> None:
        text = browser.page_nodes(self.conn, {})
        self.assertIn("лимит ElevenLabs", text)
        self.assertIn("example.com/limits", text)
        self.assertIn(">web<", text)

    def test_nodes_page_escapes_html(self) -> None:
        text = browser.page_nodes(self.conn, {})
        self.assertNotIn("<script>alert(1)</script>", text)
        self.assertIn("&lt;script&gt;", text)

    def test_nodes_filter_by_origin(self) -> None:
        text = browser.page_nodes(self.conn, {"origin": ["web"]})
        self.assertIn("лимит ElevenLabs", text)
        self.assertNotIn("опасное имя", text)

    def test_nodes_fts_search(self) -> None:
        text = browser.page_nodes(self.conn, {"q": ["лимитом"]})
        self.assertIn("лимит ElevenLabs", text)

    def test_node_detail_shows_source_and_data(self) -> None:
        text = browser.page_node(self.conn, 1)
        self.assertIn("Текст факта", text)
        self.assertIn("https://example.com/limits", text)
        self.assertIn("Сырой data", text)

    def test_unknown_node_is_a_normal_page(self) -> None:
        self.assertIn("не найден", browser.page_node(self.conn, 9999))

    def test_episodes_search(self) -> None:
        text = browser.page_episodes(self.conn, {"q": ["prolight"]})
        self.assertIn("prolight", text)
        self.assertIn(">human<", text)

    def test_episodes_filter_by_speaker(self) -> None:
        text = browser.page_episodes(self.conn, {"speaker": ["human"]})
        self.assertIn("Про память", text)
        self.assertNotIn("Запомнила про prolight", text)

    def test_pending_page(self) -> None:
        text = browser.page_pending(self.conn, {})
        self.assertIn("кандидат", text)
        self.assertIn("0.80", text)
        self.assertIn("memory_pending.py", text)

    def test_sources_page(self) -> None:
        text = browser.page_sources(self.conn)
        self.assertIn("note.md", text)
        self.assertIn("https://example.com", text)

    def test_empty_database_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = MemoryStore(Path(tmp) / "empty.db")
            empty.stats()  # файл создаётся при первом обращении
            empty.close()
            conn = browser.connect_ro(Path(tmp) / "empty.db")
            try:
                for page in (browser.page_overview(conn, Path(tmp) / "empty.db"),
                             browser.page_nodes(conn, {}),
                             browser.page_episodes(conn, {}),
                             browser.page_pending(conn, {}),
                             browser.page_sources(conn)):
                    self.assertIn("<html", page)
            finally:
                conn.close()
                close_store()


class HttpTest(Base):
    def setUp(self) -> None:
        super().setUp()
        self.httpd = browser.serve(self.db_path, "127.0.0.1", 0, token="secret")
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=3)
        super().tearDown()

    def get(self, path: str) -> tuple[int, str, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            return response.status, body, dict(response.getheaders())
        finally:
            conn.close()

    def test_token_is_required(self) -> None:
        status, body, _ = self.get("/")
        self.assertEqual(status, 401)
        self.assertIn("401", body)

    def test_token_in_query_sets_cookie(self) -> None:
        status, body, headers = self.get("/?token=secret")
        self.assertEqual(status, 200)
        self.assertIn("Обзор", body)
        self.assertIn("memtoken", headers.get("Set-Cookie", ""))

    def test_post_is_rejected(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("POST", "/nodes")
            response = conn.getresponse()
            self.assertEqual(response.status, 405)
            response.read()
        finally:
            conn.close()

    def test_token_is_not_written_to_the_log(self) -> None:
        import io
        from contextlib import redirect_stderr

        buffer = io.StringIO()
        with redirect_stderr(buffer):
            self.get("/?token=secret")
        self.assertNotIn("secret", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

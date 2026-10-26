"""Миграция схемы v1 → v2 и путь «кандидат → факт».

Живая база уже на v1, поэтому проверяем именно апгрейд существующего файла,
а не только создание нового: колонки разбора должны появиться без потери
данных, а повторное открытие не должно пытаться добавить их снова.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import memory_store as ms
from server.memory_store import MemoryStore, close_store

SCRIPT = ROOT / "scripts" / "memory_pending.py"
_spec = importlib.util.spec_from_file_location("memory_pending", SCRIPT)
pending_script = importlib.util.module_from_spec(_spec)
sys.modules["memory_pending"] = pending_script
_spec.loader.exec_module(pending_script)  # type: ignore[union-attr]


class MigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "memory.db"

    def tearDown(self) -> None:
        close_store()
        self._tmp.cleanup()

    def make_v1_database(self) -> None:
        """Собрать файл ровно в том виде, в каком он лежит в проде."""
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(ms._SCHEMA_V1)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', '1')"
        )
        conn.execute(
            "INSERT INTO pending_facts(payload, created_at) VALUES(?, 1)",
            (json.dumps({"kind": "fact", "name": "старый", "text": "кандидат"}),
             ),
        )
        conn.commit()
        conn.close()

    def columns(self, store: MemoryStore) -> set[str]:
        return {
            str(row["name"])
            for row in store.connection.execute(
                "PRAGMA table_info(pending_facts)"
            )
        }

    def test_v1_file_is_upgraded_in_place(self) -> None:
        self.make_v1_database()
        store = MemoryStore(self.db_path)
        try:
            self.assertEqual(store.schema_version, 2)
            cols = self.columns(store)
            self.assertIn("node_id", cols)
            self.assertIn("reviewed_at", cols)
            rows = store.list_pending()
            self.assertEqual(len(rows), 1, "данные не должны потеряться")
            self.assertIsNone(rows[0]["node_id"])
        finally:
            store.close()

    def test_reopening_a_v2_file_changes_nothing(self) -> None:
        self.make_v1_database()
        MemoryStore(self.db_path).close()
        again = MemoryStore(self.db_path)
        try:
            self.assertEqual(again.schema_version, 2)
            self.assertIn("node_id", self.columns(again))
        finally:
            again.close()

    def test_fresh_database_has_review_columns(self) -> None:
        store = MemoryStore(self.db_path)
        try:
            self.assertEqual(store.schema_version, 2)
            self.assertIn("reviewed_at", self.columns(store))
        finally:
            store.close()


class PendingReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "memory.db")

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        self._tmp.cleanup()

    def add_pending(self, name: str = "лимит ElevenLabs",
                    confidence: float = 0.8, text: str = "10000 символов") -> int:
        cur = self.store.connection.execute(
            "INSERT INTO pending_facts(chat_id, payload, reason, status, created_at) "
            "VALUES(?,?,?, 'pending', 1)",
            ("1", json.dumps({"kind": "fact", "name": name, "text": text,
                              "confidence": confidence}, ensure_ascii=False),
             "extractor"),
        )
        self.store.connection.commit()
        return int(cur.lastrowid)

    def test_accept_creates_node_and_marks_pending(self) -> None:
        pending_id = self.add_pending()
        node_id = pending_script.accept(self.store, pending_id)
        row = self.store.connection.execute(
            "SELECT * FROM pending_facts WHERE id=?", (pending_id,)
        ).fetchone()
        self.assertEqual(row["status"], "accepted")
        self.assertEqual(row["node_id"], node_id)
        self.assertIsNotNone(row["reviewed_at"])
        node = self.store.get_node(node_id)
        self.assertEqual(node["status"], "active")
        self.assertEqual(node["origin"], "operator")
        self.assertAlmostEqual(node["confidence"], 0.8)

    def test_accept_uses_edited_name_and_text(self) -> None:
        pending_id = self.add_pending()
        node_id = pending_script.accept(
            self.store, pending_id, name="лимит озвучки", text="10000 в месяц"
        )
        node = self.store.get_node(node_id)
        self.assertEqual(node["name"], "лимит озвучки")
        self.assertIn("10000", node["node_text"])

    def test_accepted_fact_is_searchable_and_in_prompt_path(self) -> None:
        pending_id = self.add_pending(name="лимит ElevenLabs", text="10000 символов")
        pending_script.accept(self.store, pending_id)
        from server.memory_search import search_nodes

        self.assertEqual(len(search_nodes(self.store, "лимит озвучки")), 1)

    def test_reject_marks_without_creating_a_node(self) -> None:
        pending_id = self.add_pending()
        self.assertTrue(pending_script.reject(self.store, pending_id))
        self.assertEqual(self.store.stats()["nodes"], 0)
        row = self.store.connection.execute(
            "SELECT status, node_id FROM pending_facts WHERE id=?", (pending_id,)
        ).fetchone()
        self.assertEqual(row["status"], "rejected")
        self.assertIsNone(row["node_id"])

    def test_second_review_is_refused(self) -> None:
        pending_id = self.add_pending()
        pending_script.accept(self.store, pending_id)
        with self.assertRaises(pending_script.MemoryWriteError):
            pending_script.accept(self.store, pending_id)

    def test_secret_candidate_is_still_refused_on_accept(self) -> None:
        pending_id = self.add_pending(
            name="ключ", text="token: abcdef1234567890"
        )
        with self.assertRaises(pending_script.MemoryWriteError):
            pending_script.accept(self.store, pending_id)
        row = self.store.connection.execute(
            "SELECT status FROM pending_facts WHERE id=?", (pending_id,)
        ).fetchone()
        self.assertEqual(row["status"], "pending", "кандидат не должен исчезнуть")

    def test_list_returns_pending_only(self) -> None:
        first = self.add_pending()
        second = self.add_pending(name="второй")
        pending_script.accept(self.store, first)
        left = [row["id"] for row in self.store.list_pending()]
        self.assertEqual(left, [second])


if __name__ == "__main__":
    unittest.main()

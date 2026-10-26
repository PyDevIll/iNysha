"""Схема долговременной памяти (SQLite v1): миграция, ограничения, FTS.

Каждый тест работает в своём временном каталоге: рабочая data/ проекта
не открывается и не создаётся (иначе тестовая запись попала бы в живую
память агента).

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.memory_store import SCHEMA_VERSION, MemoryStore, close_store, get_store, now_ts

TABLES = (
    "meta",
    "episodes",
    "nodes",
    "node_aliases",
    "edges",
    "edge_sources",
    "memory_index",
    "pending_facts",
    "nodes_fts",
    "episodes_fts",
)


class StoreTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.db_path = self.data_dir / "memory.db"
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        self._tmp.cleanup()

    # ── helpers ────────────────────────────────────────────────────
    def add_node(self, name: str = "Сервер", kind: str = "server", **kw) -> int:
        now = now_ts()
        cols = {
            "kind": kind,
            "name": name,
            "name_norm": kw.pop("name_norm", name.lower()),
            "node_text": kw.pop("node_text", ""),
            "lemma_text": kw.pop("lemma_text", ""),
            "scope": kw.pop("scope", "shared"),
            "origin": kw.pop("origin", "operator"),
            "created_at": now,
            "updated_at": now,
            **kw,
        }
        placeholders = ", ".join("?" for _ in cols)
        cur = self.store.connection.execute(
            f"INSERT INTO nodes({', '.join(cols)}) VALUES({placeholders})",
            tuple(cols.values()),
        )
        self.store.connection.commit()
        return int(cur.lastrowid)

    def add_episode(self, text: str = "привет", **kw) -> int:
        cols = {
            "chat_id": kw.pop("chat_id", "1"),
            "speaker": kw.pop("speaker", "human"),
            "kind": kw.pop("kind", "message"),
            "ts": kw.pop("ts", now_ts()),
            "text": text,
            "lemma_text": kw.pop("lemma_text", ""),
            "hash": kw.pop("hash", "h1"),
            **kw,
        }
        placeholders = ", ".join("?" for _ in cols)
        cur = self.store.connection.execute(
            f"INSERT INTO episodes({', '.join(cols)}) VALUES({placeholders})",
            tuple(cols.values()),
        )
        self.store.connection.commit()
        return int(cur.lastrowid)

    def search(self, table: str, query: str) -> list[int]:
        rows = self.store.connection.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ?", (query,)
        ).fetchall()
        return [int(r["rowid"]) for r in rows]


class SchemaTest(StoreTestBase):
    def test_store_is_lazy(self) -> None:
        """Одна лишь ссылка на MemoryStore не создаёт файл и не читает data/."""
        fresh = self.data_dir / "lazy" / "memory.db"
        MemoryStore(fresh)
        self.assertFalse(fresh.exists())

    def test_schema_creates_all_tables(self) -> None:
        names = {
            str(row["name"])
            for row in self.store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        }
        for table in TABLES:
            self.assertIn(table, names)

    def test_schema_version_recorded(self) -> None:
        self.assertEqual(self.store.schema_version, SCHEMA_VERSION)
        row = self.store.connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        self.assertEqual(int(row["value"]), SCHEMA_VERSION)

    def test_wal_mode_enabled(self) -> None:
        mode = self.store.connection.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")

    def test_foreign_keys_enabled(self) -> None:
        value = self.store.connection.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]
        self.assertEqual(int(value), 1)

    def test_reopen_does_not_reapply_or_fail(self) -> None:
        self.add_node()
        self.store.close()
        again = MemoryStore(self.db_path)
        self.assertEqual(again.schema_version, SCHEMA_VERSION)
        self.assertEqual(again.stats()["nodes"], 1)
        again.close()

    def test_newer_schema_is_refused(self) -> None:
        self.store.connection  # noqa: B018 - create the schema first
        self.store.close()
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'",
            (str(SCHEMA_VERSION + 5),),
        )
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError):
            MemoryStore(self.db_path).connection

    def test_integrity_check(self) -> None:
        self.assertTrue(self.store.integrity_ok())

    def test_database_is_not_world_readable(self) -> None:
        self.store.connection  # noqa: B018 - create the file
        mode = self.db_path.stat().st_mode & 0o777
        self.assertEqual(mode & 0o077, 0, oct(mode))

    def test_stats_counts_rows(self) -> None:
        self.add_node()
        self.add_episode()
        stats = self.store.stats()
        self.assertEqual(stats["nodes"], 1)
        self.assertEqual(stats["episodes"], 1)


class ConstraintTest(StoreTestBase):
    def test_episode_duplicate_is_rejected(self) -> None:
        self.add_episode(ts=100, hash="abc")
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_episode(ts=100, hash="abc")

    def test_episode_same_text_new_timestamp_allowed(self) -> None:
        self.add_episode(ts=100, hash="abc")
        self.add_episode(ts=200, hash="abc")
        self.assertEqual(self.store.stats()["episodes"], 2)

    def test_episode_speaker_is_checked(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_episode(speaker="robot")

    def test_episode_tone_is_checked(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_episode(tone="joky")

    def test_node_uniqueness_includes_kind_and_scope(self) -> None:
        self.add_node("Андрей", kind="person")
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_node("Андрей", kind="person")
        # Другой scope — другое понятие.
        self.add_node("Андрей", kind="person", scope="chat:1")
        # Другой kind — другое понятие.
        self.add_node("Андрей", kind="topic")

    def test_node_data_must_be_valid_json(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_node(data="{not json")

    def test_node_origin_is_checked(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_node(origin="guess")

    def test_node_confidence_range(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_node(confidence=1.5)

    def test_alias_unique_per_node(self) -> None:
        node_id = self.add_node()
        conn = self.store.connection
        conn.execute(
            "INSERT INTO node_aliases(node_id, alias, alias_norm) VALUES(?,?,?)",
            (node_id, "сервак", "сервак"),
        )
        conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO node_aliases(node_id, alias, alias_norm) VALUES(?,?,?)",
                (node_id, "сервак", "сервак"),
            )

    def test_alias_cascades_with_node(self) -> None:
        node_id = self.add_node()
        conn = self.store.connection
        conn.execute(
            "INSERT INTO node_aliases(node_id, alias, alias_norm) VALUES(?,?,?)",
            (node_id, "сервак", "сервак"),
        )
        conn.commit()
        conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        conn.commit()
        left = conn.execute("SELECT count(*) FROM node_aliases").fetchone()[0]
        self.assertEqual(left, 0)

    def test_edge_rejects_self_loop(self) -> None:
        node = self.add_node()
        with self.assertRaises(sqlite3.IntegrityError):
            self._add_edge(node, node)

    def test_edge_rejects_inverted_validity(self) -> None:
        a, b = self.add_node("A"), self.add_node("B")
        with self.assertRaises(sqlite3.IntegrityError):
            self._add_edge(a, b, valid_from=200, valid_to=100)

    def test_edge_relation_is_checked(self) -> None:
        a, b = self.add_node("A"), self.add_node("B")
        with self.assertRaises(sqlite3.IntegrityError):
            self._add_edge(a, b, rel="hates")

    def test_edge_is_idempotent_per_source_time(self) -> None:
        a, b = self.add_node("A"), self.add_node("B")
        self._add_edge(a, b, valid_from=100)
        with self.assertRaises(sqlite3.IntegrityError):
            self._add_edge(a, b, valid_from=100)

    def test_edge_cannot_reference_missing_node(self) -> None:
        a = self.add_node("A")
        with self.assertRaises(sqlite3.IntegrityError):
            self._add_edge(a, 9999)

    def test_pending_fact_requires_json_payload(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "INSERT INTO pending_facts(payload, created_at) VALUES(?,?)",
                ("nope", now_ts()),
            )

    def _add_edge(
        self,
        src: int,
        dst: int,
        rel: str = "likes",
        valid_from: int = 100,
        valid_to: int | None = None,
    ) -> None:
        self.store.connection.execute(
            "INSERT INTO edges(src, dst, rel, valid_from, valid_to) "
            "VALUES(?,?,?,?,?)",
            (src, dst, rel, valid_from, valid_to),
        )
        self.store.connection.commit()


class FtsTriggerTest(StoreTestBase):
    def test_inserted_node_is_searchable_by_lemmas(self) -> None:
        node_id = self.add_node(
            "Сервер", kind="server", lemma_text="сервер упасть дважды"
        )
        self.assertEqual(self.search("nodes_fts", '"сервер"'), [node_id])
        self.assertEqual(self.search("nodes_fts", '"упасть"'), [node_id])

    def test_updated_node_leaves_stale_term_behind(self) -> None:
        node_id = self.add_node("Сервер", kind="server", lemma_text="сервер упасть")
        conn = self.store.connection
        conn.execute(
            "UPDATE nodes SET lemma_text=?, updated_at=? WHERE id=?",
            ("сервер перезапуск", now_ts(), node_id),
        )
        conn.commit()
        self.assertEqual(self.search("nodes_fts", '"перезапуск"'), [node_id])
        self.assertEqual(self.search("nodes_fts", '"упасть"'), [])

    def test_deleted_node_disappears_from_index(self) -> None:
        node_id = self.add_node("Сервер", kind="server", lemma_text="сервер упасть")
        conn = self.store.connection
        conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        conn.commit()
        self.assertEqual(self.search("nodes_fts", '"сервер"'), [])

    def test_inserted_episode_is_searchable(self) -> None:
        ep_id = self.add_episode(
            "Сервер упал", lemma_text="сервер упасть"
        )
        self.assertEqual(self.search("episodes_fts", '"сервер"'), [ep_id])

    def test_episode_summary_is_searchable(self) -> None:
        ep_id = self.add_episode(
            "…", lemma_text="", summary="обсуждали перезапуск"
        )
        self.assertEqual(
            self.search("episodes_fts", '"обсуждали"'), [ep_id]
        )

    def test_updated_episode_refreshes_index(self) -> None:
        ep_id = self.add_episode("старое", lemma_text="старый")
        conn = self.store.connection
        conn.execute(
            "UPDATE episodes SET text=?, lemma_text=? WHERE id=?",
            ("новое", "новый", ep_id),
        )
        conn.commit()
        self.assertEqual(self.search("episodes_fts", '"новый"'), [ep_id])
        self.assertEqual(self.search("episodes_fts", '"старый"'), [])

    def test_deleted_episode_disappears_from_index(self) -> None:
        ep_id = self.add_episode("старое", lemma_text="старый")
        conn = self.store.connection
        conn.execute("DELETE FROM episodes WHERE id=?", (ep_id,))
        conn.commit()
        self.assertEqual(self.search("episodes_fts", '"старый"'), [])


class SingletonTest(StoreTestBase):
    def test_get_store_is_cached_per_path(self) -> None:
        first = get_store(self.db_path)
        second = get_store(self.db_path)
        self.assertIs(first, second)
        self.assertEqual(first.path, self.db_path)

    def test_different_paths_are_different_stores(self) -> None:
        first = get_store(self.db_path)
        other = get_store(self.data_dir / "other.db")
        self.assertIsNot(first, other)

    def test_close_store_resets_singleton(self) -> None:
        first = get_store(self.db_path)
        close_store(self.db_path)
        second = get_store(self.db_path)
        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()

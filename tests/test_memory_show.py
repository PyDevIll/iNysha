"""Просмотр блока памяти: тот же отбор, что в проде, но без записи метрик.

Проверяем:
  * `explain_block` не трогает `memory_index` и не пишет в живую базу;
  * причины попадания/выпадения видны по каждому элементу (бюджет, квота,
    изменившийся файл);
  * отчёт печатает и разбор, и сам блок в том виде, в каком его увидит модель;
  * снимок базы делается из read-only соединения и не меняет оригинал.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.memory_store import MemoryStore, close_store

SCRIPT = ROOT / "scripts" / "memory_show.py"
_spec = importlib.util.spec_from_file_location("memory_show", SCRIPT)
memory_show = importlib.util.module_from_spec(_spec)
sys.modules["memory_show"] = memory_show
_spec.loader.exec_module(memory_show)  # type: ignore[union-attr]


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.upsert_node(
            kind="fact", name="лимит ElevenLabs", name_norm="лимит elevenlabs",
            node_text="10000 символов в месяц",
            data=json.dumps({"text": "10000 символов в месяц"},
                            ensure_ascii=False),
            origin="operator", confidence=0.9,
        )
        self.store.add_episode(
            chat_id="1", speaker="human", kind="message", ts=1_790_000_000,
            text="Помни, что лимит озвучки кончается в конце месяца",
        )

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        self._tmp.cleanup()

    def explain(self, query: str = "лимит озвучки", **kwargs) -> dict:
        self.store.close()
        return memory_show.explain(self.db_path, query, **kwargs)


class ExplainTest(Base):
    def test_block_is_built_from_a_snapshot(self) -> None:
        result = self.explain()
        self.assertIn("## Из памяти", result["block"])
        self.assertIn("лимит ElevenLabs", result["block"])
        self.assertGreater(result["tokens"], 0)

    def test_trace_marks_included_items(self) -> None:
        result = self.explain()
        ok = [entry for entry in result["trace"] if entry["reason"] == "ok"]
        self.assertTrue(ok)
        self.assertTrue(all(entry["kind"] in ("node", "episode") for entry in ok))

    def test_usage_is_not_recorded(self) -> None:
        self.explain()
        reopened = MemoryStore(self.db_path)
        try:
            rows = reopened.connection.execute(
                "SELECT count(*) c FROM memory_index"
            ).fetchone()["c"]
            self.assertEqual(int(rows), 0)
        finally:
            reopened.close()

    def test_external_quota_is_reported(self) -> None:
        for index in range(4):
            self.store.upsert_node(
                kind="fact", name=f"внешний факт {index}",
                name_norm=f"внешний факт {index}",
                node_text=f"данные {index}",
                data=json.dumps({"text": f"данные {index}"}, ensure_ascii=False),
                origin="web", confidence=0.6,
            )
        result = self.explain("внешний факт")
        reasons = [entry["reason"] for entry in result["trace"]]
        self.assertEqual(reasons.count("external_quota"), 1)

    def test_changed_file_is_reported_and_hidden(self) -> None:
        note = Path(self._tmp.name) / "note.md"
        note.write_text("правило", encoding="utf-8")
        stat = os.stat(note)
        self.store.upsert_node(
            kind="rule", name="правило из файла", name_norm="правило из файла",
            node_text="правило из файла",
            data=json.dumps({
                "text": "правило из файла",
                "source": {"kind": "file", "ref": str(note),
                           "mtime": int(stat.st_mtime_ns)},
            }, ensure_ascii=False),
            origin="tool", confidence=0.6,
        )
        future = int(time.time()) + 120
        os.utime(note, (future, future))
        result = self.explain("правило из файла")
        reasons = [entry["reason"] for entry in result["trace"]]
        self.assertIn("file_changed", reasons)
        self.assertNotIn("правило из файла", result["block"])

    def test_budget_can_drop_everything(self) -> None:
        result = self.explain("лимит озвучки", budget=8)
        self.assertEqual(result["block"], "")
        self.assertTrue(any(entry["reason"] == "budget"
                            for entry in result["trace"]))


class ReportTest(Base):
    def test_report_shows_block_and_summary(self) -> None:
        text = memory_show.format_report(self.explain(), show_items=False)
        self.assertIn("запрос:", text)
        self.assertIn("бюджет:", text)
        self.assertIn("## Из памяти", text)
        self.assertNotIn("→ в блоке", text)

    def test_items_show_scores_and_reasons(self) -> None:
        text = memory_show.format_report(self.explain(), show_items=True)
        self.assertIn("→ в блоке", text)
        self.assertIn("score=", text)
        self.assertIn("conf=", text)
        self.assertIn("итого в блоке:", text)

    def test_empty_result_is_explicit(self) -> None:
        result = self.explain("совершенно посторонний запрос про кулинарию")
        text = memory_show.format_report(result, show_items=True)
        self.assertTrue("пусто" in text or "ничего не найдено" in text)


class SnapshotTest(Base):
    def test_snapshot_does_not_modify_the_original(self) -> None:
        self.store.close()
        before = self.db_path.stat().st_size
        with tempfile.TemporaryDirectory() as tmp:
            copy_path = memory_show.snapshot(self.db_path, tmp)
            self.assertTrue(copy_path.exists())
            copy = MemoryStore(copy_path)
            try:
                self.assertGreaterEqual(copy.stats()["episodes"], 1)
            finally:
                copy.close()
        self.assertEqual(self.db_path.stat().st_size, before)


if __name__ == "__main__":
    unittest.main()

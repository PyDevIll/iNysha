"""Логирование: память отлаживается отдельно, контекст — без шума.

Проверяем то, что нельзя увидеть глазами: что `log_state` снова не начнёт
писать 13 строк на каждый вызов модели, что один и тот же разрыв
tool-последовательности не рапортуется каждые 30 секунд, и что токены
загрузки MAX никогда не попадают в лог.

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

from loguru import logger

from context_manager import ContextPool, repair_tool_sequence
from server.memory_text import fts_query, preview


class Capture:
    """Собрать записи loguru без файлов и без изменения глобального уровня."""

    def __init__(self, level: str = "TRACE"):
        self.records: list = []
        self._id = logger.add(
            lambda message: self.records.append(message.record),
            level=level,
            format="{message}",
        )

    def __enter__(self) -> "Capture":
        return self

    def __exit__(self, *exc_info) -> None:
        logger.remove(self._id)

    def messages(self) -> list[str]:
        return [str(r["message"]) for r in self.records]

    def levels(self) -> list[str]:
        return [r["level"].name for r in self.records]

    def count(self, level: str) -> int:
        return sum(1 for name in self.levels() if name == level)


class ContextLogStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.pool = ContextPool(data_dir=Path(self._tmp.name))
        self.pool.append({"role": "user", "name": "User", "content": "привет"})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_state_is_one_debug_line(self) -> None:
        # memory.log/agent.log слушают с DEBUG: детали на TRACE туда не попадут.
        with Capture(level="DEBUG") as cap:
            self.pool.log_state("TEST")
        self.assertEqual(len(cap.messages()), 1, cap.messages())
        self.assertEqual(cap.levels(), ["DEBUG"])
        line = cap.messages()[0]
        for field in ("entries=", "boundary=", "assembled=", "overflow="):
            self.assertIn(field, line)

    def test_details_survive_at_trace(self) -> None:
        with Capture(level="TRACE") as cap:
            self.pool.log_state("TEST")
        self.assertEqual(len(cap.messages()), 3, cap.messages())

    def test_detailed_state_stays_available(self) -> None:
        with Capture() as cap:
            self.pool.log_state("TEST", detailed=True)
        self.assertGreaterEqual(len(cap.messages()), 2, cap.messages())
        self.assertIn("INFO", cap.levels())
        joined = "\n".join(cap.messages())
        self.assertIn("first=", joined)

    def test_no_info_noise_per_turn(self) -> None:
        """Ни одной INFO-строки на ход: иначе 85 вызовов снова дают 1105 строк."""
        with Capture(level="INFO") as cap:
            self.pool.log_state("TEST")
        self.assertEqual(cap.messages(), [])


class RepairLogTest(unittest.TestCase):
    @staticmethod
    def damaged(call_id: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "exec_python", "arguments": "{}"},
                }],
            },
            {"role": "user", "content": "продолжай"},
        ]

    def test_same_damage_reported_once(self) -> None:
        with Capture() as cap:
            repair_tool_sequence(self.damaged("call-once-1"))
            repair_tool_sequence(self.damaged("call-once-1"))
            repair_tool_sequence(self.damaged("call-once-1"))
        self.assertEqual(cap.count("WARNING"), 1, cap.messages())
        # Повторы видны, но не шумят: одна DEBUG-строка на каждый.
        self.assertEqual(
            len([m for m in cap.messages() if "reported earlier" in m]), 2
        )

    def test_new_damage_warns_again(self) -> None:
        with Capture() as cap:
            repair_tool_sequence(self.damaged("call-new-1"))
            repair_tool_sequence(self.damaged("call-new-2"))
        self.assertEqual(cap.count("WARNING"), 2, cap.messages())

    def test_repair_still_synthesises_result(self) -> None:
        repaired = repair_tool_sequence(self.damaged("call-once-shape"))
        roles = [m["role"] for m in repaired]
        self.assertEqual(roles, ["assistant", "tool", "user"])
        self.assertEqual(repaired[1]["tool_call_id"], "call-once-shape")


class MemoryLogTest(unittest.TestCase):
    def test_fts_query_is_trace_only(self) -> None:
        """Поиск идёт каждый ход: на DEBUG он засорил бы memory.log."""
        with Capture(level="DEBUG") as cap:
            fts_query("сервером упал")
        self.assertEqual(
            [m for m in cap.messages() if "fts_query" in m], []
        )

    def test_preview_truncates_and_marks_length(self) -> None:
        text = "слово " * 40
        shown = preview(text, 30)
        self.assertLess(len(shown), len(text))
        self.assertIn("chars", shown)
        self.assertEqual(preview("коротко"), "коротко")
        self.assertEqual(preview(None), "")

    def test_morphology_failure_warns_once(self) -> None:
        import server.memory_text as mt

        original = (mt._morph, mt._morph_attempted)
        mt._morph, mt._morph_attempted = None, False
        real_module = sys.modules.get("pymorphy3")

        class Broken:
            def __init__(self, *a, **kw):
                raise RuntimeError("no dictionaries")

        try:
            import pymorphy3

            saved = pymorphy3.MorphAnalyzer
            pymorphy3.MorphAnalyzer = Broken
            try:
                with Capture() as cap:
                    self.assertFalse(mt.has_morphology())
                    self.assertFalse(mt.has_morphology())
                    self.assertEqual(mt.lemma("сервер"), "сервер")
            finally:
                pymorphy3.MorphAnalyzer = saved
            warnings = [m for m in cap.messages() if "pymorphy3" in m]
            self.assertEqual(len(warnings), 1, warnings)
        finally:
            mt._morph, mt._morph_attempted = None, False
            if real_module is None:  # pragma: no cover - morphology is installed
                sys.modules.pop("pymorphy3", None)
            mt.has_morphology()


class MaxLogRedactionTest(unittest.TestCase):
    def test_upload_token_is_masked(self) -> None:
        from lib.max_bot import _mask_secret

        self.assertEqual(_mask_secret("abc123"), "<set len=6>")
        self.assertNotIn("abc123", _mask_secret("abc123"))

    def test_nested_token_is_redacted(self) -> None:
        from lib.max_bot import _redact

        payload = {
            "chat_id": 1,
            "attachments": [
                {"type": "file", "payload": {"token": "SECRET-TOKEN", "url": "u"}}
            ],
            "nested": [{"access_token": "ANOTHER"}],
        }
        cleaned = str(_redact(payload))
        self.assertNotIn("SECRET-TOKEN", cleaned)
        self.assertNotIn("ANOTHER", cleaned)
        self.assertIn("chat_id", cleaned)

    def test_brief_truncates_bodies(self) -> None:
        from lib.max_bot import _brief

        long_body = "x" * 1000
        self.assertLess(len(_brief(long_body)), 300)
        self.assertIn("chars", _brief(long_body))
        self.assertEqual(_brief({"a": 1}), "{'a': 1}")


if __name__ == "__main__":
    unittest.main()

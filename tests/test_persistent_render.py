"""Рендер профиля памяти: без Python-repr, с бюджетом и без потери фактов.

До этого `as_context_string()` склеивал `- **{k}**: {v}`, где `v` — вложенный
dict/list, то есть в промпт уезжал repr: 10.5 КБ / 4408 токенов на каждый ход.
Здесь проверяем компактную форму, детерминизм (префикс-кэш) и то, что при
превышении бюджета блоки режутся целиком, а не посреди заголовка.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_manager import (
    PERSISTENT_FILE,
    PersistentStore,
    _token_count,
    render_persistent,
)

SAMPLE = {
    "facts": {
        "operator": {
            "chat_id": "326963375",
            "notes": ["любит спорить", "просил не менять тон"],
        },
        "date": "18.10.2026 воскресенье",
    },
    "lessons_learned": [
        {"topic": "ffmpeg", "detail": "установлен в venv", "date": "18.10.2026"},
        {"topic": "fs_edit", "detail": "по номерам строк ломается"},
    ],
    "autonomy": {"дан": "21.10.2026 — полная автономность"},
}


class RenderShapeTest(unittest.TestCase):
    def test_no_python_repr_left(self) -> None:
        text = render_persistent(SAMPLE, max_tokens=10**6)
        for artifact in ("{'", "}", "['", "True", "None"):
            self.assertNotIn(artifact, text, f"{artifact!r} в {text!r}")

    def test_nested_list_values_survive(self) -> None:
        text = render_persistent(SAMPLE, max_tokens=10**6)
        self.assertIn("chat_id: 326963375", text)
        self.assertIn("любит спорить", text)
        self.assertIn("просил не менять тон", text)

    def test_list_of_dicts_renders_one_line_each(self) -> None:
        text = render_persistent(SAMPLE, max_tokens=10**6)
        self.assertIn("установлен в venv", text)
        self.assertIn("по номерам строк ломается", text)

    def test_depth_does_not_explode(self) -> None:
        deep = {"a": {"b": {"c": {"d": ["x"]}}, "e": [{"f": "y"}]}}
        text = render_persistent(deep, max_tokens=10**6)
        self.assertLessEqual(max(len(line) for line in text.splitlines()), 300)

    def test_empty_profile_is_empty(self) -> None:
        self.assertEqual(render_persistent({}), "")

    def test_bool_and_none_are_readable(self) -> None:
        text = render_persistent({"flags": {"on": True, "off": False, "no": None}},
                                 max_tokens=10**6)
        self.assertIn("on: да", text)
        self.assertIn("off: нет", text)
        self.assertIn("no: —", text)

    def test_long_value_is_capped(self) -> None:
        text = render_persistent({"big": "я" * 1000}, max_tokens=10**6)
        self.assertLess(len(text), 500)
        self.assertIn("симв.", text)


class DeterminismTest(unittest.TestCase):
    def test_same_input_same_bytes(self) -> None:
        """Префикс-кэш: неизменный профиль обязан давать идентичный блок."""
        first = render_persistent(SAMPLE, max_tokens=10**6)
        second = render_persistent(json.loads(json.dumps(SAMPLE)), max_tokens=10**6)
        self.assertEqual(first, second)

    def test_order_follows_json(self) -> None:
        data = {"z": "1", "a": "2"}
        self.assertLess(
            render_persistent(data).index("**z**"),
            render_persistent(data).index("**a**"),
        )


class BudgetTest(unittest.TestCase):
    def test_budget_is_respected(self) -> None:
        for budget in (30, 45, 60, 90):
            rendered = render_persistent(SAMPLE, max_tokens=budget)
            self.assertLessEqual(_token_count(rendered), budget, rendered)

    def test_trim_marker_counts_blocks(self) -> None:
        rendered = render_persistent(SAMPLE, max_tokens=45)
        self.assertIn("обрезано", rendered)
        self.assertIn("блок", rendered)
        self.assertIn(PERSISTENT_FILE, rendered)

    def test_no_dangling_header(self) -> None:
        """Каждый оставленный заголовок блока обязан иметь детей."""
        for budget in (20, 30, 45, 60, 90, 200):
            rendered = render_persistent(SAMPLE, max_tokens=budget)
            lines = [line for line in rendered.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                if line.startswith("- **") and line.endswith("**"):
                    self.assertLess(index + 1, len(lines), rendered)
                    self.assertTrue(
                        lines[index + 1].startswith("  - "),
                        f"висячий заголовок при budget={budget}:\n{rendered}",
                    )

    def test_big_budget_keeps_everything(self) -> None:
        rendered = render_persistent(SAMPLE, max_tokens=10**6)
        self.assertNotIn("обрезано", rendered)


class PersistentStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_store_renders_compact_block(self) -> None:
        (self.data_dir / PERSISTENT_FILE).write_text(
            json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8"
        )
        store = PersistentStore(self.data_dir)
        text = store.as_context_string()
        self.assertTrue(text.startswith("## Persistent Memory"))
        self.assertNotIn("{'", text)

    def test_missing_file_renders_nothing(self) -> None:
        self.assertEqual(PersistentStore(self.data_dir).as_context_string(), "")

    def test_broken_file_does_not_crash(self) -> None:
        (self.data_dir / PERSISTENT_FILE).write_text("{broken", encoding="utf-8")
        self.assertEqual(PersistentStore(self.data_dir).as_context_string(), "")


if __name__ == "__main__":
    unittest.main()

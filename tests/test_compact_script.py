"""Скрипт компакции профиля: ничего не теряет, делает бэкап, идемпотентен.

Главный риск этапа — «сжали и забыли»: любая строка памяти должна остаться
в файле, на который указывает JSON. Поэтому тест сверяет все скалярные
значения исходника с текстом созданных .md.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "compact_persistent_memory.py"
_spec = importlib.util.spec_from_file_location("compact_persistent_memory", SCRIPT)
compact_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compact_script)  # type: ignore[union-attr]

PROFILE = {
    "facts": {"operator": {"chat_id": "1", "notes": ["любит спорить"]}},
    "ideas_from_andrey": [
        {"id": 1, "title": "Портал", "essence": "живой тестер", "verdict": "спорно"}
    ],
    "lessons_learned": [
        {"topic": "ffmpeg", "detail": "поставлен в venv", "date": "18.10.2026"}
    ],
    "lore_parad_zvezd": {"owner": "Денис", "world": "Пелагон"},
    "virtual_machines": {"servers": [{"id": "self", "ip": "2.24.160.163"}]},
    "autonomy": {"дан": "полная автономность"},
}


def scalars(value, out: list[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            scalars(item, out)
    elif isinstance(value, list):
        for item in value:
            scalars(item, out)
    else:
        out.append(str(value))


class CompactTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.path = self.data_dir / "persistent_memory.json"
        self.path.write_text(
            json.dumps(PROFILE, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_script(self, *args: str) -> int:
        return compact_script.main(["--data-dir", str(self.data_dir), *args])

    def test_compact_keeps_unplanned_keys(self) -> None:
        new_data, files = compact_script.compact(PROFILE)
        self.assertEqual(new_data["facts"], PROFILE["facts"])
        self.assertEqual(new_data["autonomy"], PROFILE["autonomy"])
        for key in compact_script.SPLIT_PLAN:
            self.assertIsInstance(new_data[key], str)
        self.assertEqual(set(files), {plan[0] for plan in compact_script.SPLIT_PLAN.values()})

    def test_pointer_names_its_file(self) -> None:
        new_data, _ = compact_script.compact(PROFILE)
        for key, (file_name, _heading, _pointer) in compact_script.SPLIT_PLAN.items():
            self.assertIn(file_name, new_data[key])

    def test_no_value_is_lost(self) -> None:
        _, files = compact_script.compact(PROFILE)
        for key, (file_name, _h, _p) in compact_script.SPLIT_PLAN.items():
            expected: list[str] = []
            scalars(PROFILE[key], expected)
            text = files[file_name]
            for value in expected:
                self.assertIn(value, text, f"{key}/{file_name}: потеряно {value!r}")

    def test_already_compacted_value_is_not_split_again(self) -> None:
        new_data, files = compact_script.compact(PROFILE)
        again, files_again = compact_script.compact(new_data)
        self.assertEqual(again, new_data)
        self.assertEqual(files_again, {})

    def test_dry_run_writes_nothing(self) -> None:
        before = self.path.read_text(encoding="utf-8")
        self.assertEqual(self.run_script(), 0)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertFalse((self.data_dir / "lessons.md").exists())

    def test_apply_writes_files_and_backup(self) -> None:
        self.assertEqual(self.run_script("--apply"), 0)
        for file_name, _h, _p in compact_script.SPLIT_PLAN.values():
            self.assertTrue((self.data_dir / file_name).exists(), file_name)
        backups = list(self.data_dir.glob("persistent_memory.json.bak_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(
            json.loads(backups[0].read_text(encoding="utf-8")), PROFILE
        )
        compacted = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsInstance(compacted["lessons_learned"], str)

    def test_second_apply_changes_nothing(self) -> None:
        self.run_script("--apply")
        snapshot = self.path.read_text(encoding="utf-8")
        self.assertEqual(self.run_script("--apply"), 0)
        self.assertEqual(self.path.read_text(encoding="utf-8"), snapshot)

    def test_existing_md_is_not_overwritten(self) -> None:
        target = self.data_dir / "lessons.md"
        target.write_text("мои правки руками\n", encoding="utf-8")
        self.assertEqual(self.run_script("--apply"), 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "мои правки руками\n")
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8")), PROFILE
        )

    def test_missing_profile_is_reported(self) -> None:
        self.path.unlink()
        self.assertEqual(self.run_script(), 1)

    def test_stale_read_is_not_written(self) -> None:
        """Живой агент правит файл сам: устаревшее чтение не должно затирать."""
        real_read = Path.read_text
        calls = {"n": 0}

        def racing_read(self, *args, **kwargs):  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] == 2:  # первый читает main(), второй — проверка гонки
                return '{"concurrent": "write"}'
            return real_read(self, *args, **kwargs)

        with unittest.mock.patch.object(Path, "read_text", racing_read):
            code = self.run_script("--apply")

        self.assertEqual(code, 3)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), PROFILE)
        self.assertFalse((self.data_dir / "lessons.md").exists())


if __name__ == "__main__":
    unittest.main()

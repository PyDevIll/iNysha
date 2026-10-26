"""Старт сервера: подготовка памяти исполняется, а не падает на NameError.

Этот тест появился после реального сбоя: в `runtime.startup()` стоял вызов
`memory_context.enabled()` без импорта. Тесты проходили, потому что никто не
вызывал `startup()`, а сервис падал в цикле рестартов. Теперь стартовая
подготовка памяти — отдельная функция, и она вызывается здесь.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import config
from server import memory_context, runtime
from server.memory_store import MemoryStore


class StartMemoryTest(unittest.TestCase):
    def test_noop_when_backend_is_json(self) -> None:
        with mock.patch.object(config, "MEMORY_BACKEND", "json"), \
             mock.patch.object(memory_context, "warmup") as warmup:
            runtime.start_memory_components()
        warmup.assert_not_called()

    def test_warms_up_morphology_when_sqlite(self) -> None:
        with mock.patch.object(config, "MEMORY_BACKEND", "sqlite"), \
             mock.patch.object(memory_context, "warmup", return_value=True) as warmup:
            runtime.start_memory_components()
        warmup.assert_called_once_with()

    def test_real_warmup_does_not_raise(self) -> None:
        with mock.patch.object(config, "MEMORY_BACKEND", "sqlite"):
            runtime.start_memory_components()


class ModuleWiringTest(unittest.TestCase):
    """Дешёвая страховка от `NameError` в стартовом коде."""

    def test_runtime_imports_memory_context(self) -> None:
        self.assertIs(runtime.memory_context, memory_context)

    def test_memory_modules_import_cleanly(self) -> None:
        import importlib

        for name in (
            "server.memory_store",
            "server.memory_text",
            "server.memory_search",
            "server.memory_write",
            "server.memory_context",
            "server.memory_extract",
            "server.memory_episodes",
            "builtin_tools.memory_tools",
        ):
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_memory_log_sink_is_registered(self) -> None:
        """configure_logging заводит отдельный лог только для server.memory*.

        Строку пишем из самого модуля памяти: у синка есть фильтр по имени
        логгера, и «проба» из теста в memory.log не попадёт by design.
        """
        from loguru import logger

        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "agent.log"
            memory_log = Path(tmp) / "memory.log"
            try:
                with mock.patch.object(config, "LOG_FILE", log_file), \
                     mock.patch.object(config, "MEMORY_LOG_FILE", memory_log), \
                     mock.patch.object(config, "DATA_DIR", Path(tmp)):
                    runtime.configure_logging()
                    probe = MemoryStore(Path(tmp) / "probe.db")
                    try:
                        self.assertTrue(probe.integrity_ok())
                    finally:
                        probe.close()
                    logger.info("agent-side probe line")
                    logger.complete()
            finally:
                # configure_logging снёс глобальные хендлеры — вернуть хоть что-то,
                # иначе остальные тесты потеряют вывод.
                logger.remove()
                logger.add(sys.stderr, level="INFO")

            self.assertTrue(memory_log.exists(), "memory.log не создан")
            text = memory_log.read_text(encoding="utf-8")
            self.assertIn("integrity_check ok", text)
            # Фильтр синка: чужие строки агента в память не попадают.
            self.assertNotIn("agent-side probe line", text)
            agent_text = log_file.read_text(encoding="utf-8")
            self.assertIn("integrity_check", agent_text)
            self.assertIn("agent-side probe line", agent_text)


if __name__ == "__main__":
    unittest.main()

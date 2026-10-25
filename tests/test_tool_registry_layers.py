"""Тесты разделения builtin- и client-слоёв ToolRegistry.

Проверяем, что клиентские инструменты не входят в builtin-пакет, переживают
hot reload и что одинаковые имена между слоями запрещены.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.remote_tools import register_all as register_client_tools
from tool_registry import ToolRegistry


async def _client_probe() -> str:
    return "client"


class ToolRegistryLayersTest(unittest.TestCase):
    def test_local_screen_and_speaker_tools_are_not_builtins(self) -> None:
        registry = ToolRegistry()
        registry.hot_reload()

        for name in ("get_screenshot", "analyze_dynamic_scene", "tts_speak_aloud"):
            self.assertNotIn(name, registry._tools)

    def test_client_layer_survives_hot_reload(self) -> None:
        registry = ToolRegistry()
        register_client_tools(registry)

        before = set(registry.client_tool_names)
        self.assertEqual(
            before,
            {
                "list_clients",
                "get_screenshot",
                "analyze_dynamic_scene",
                "tts_speak_aloud",
                "client_record_audio",
                "client_notify",
            },
        )

        registry.hot_reload()

        self.assertEqual(set(registry.client_tool_names), before)
        self.assertEqual(
            set(registry.client_tool_names) - set(registry.tool_names),
            set(),
        )
        self.assertEqual(
            registry.get_tool("get_screenshot").func.__module__,
            "server.remote_tools",
        )

    def test_duplicate_client_name_is_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register_client_function(_client_probe, "probe", "probe")

        with self.assertRaises(ValueError):
            registry.register_client_function(_client_probe, "probe", "probe")

    def test_client_tool_cannot_override_builtin(self) -> None:
        registry = ToolRegistry()
        registry.hot_reload()

        with self.assertRaises(ValueError):
            registry.register_client_function(
                _client_probe, "tts_generate", "must not replace builtin"
            )


if __name__ == "__main__":
    unittest.main()

"""Central instance configuration: identity, paths, routes, prompt render."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class InstanceConfigTest(unittest.TestCase):
    def _run(self, extra_env: dict[str, str], code: str) -> str:
        env = os.environ.copy()
        env.update(extra_env)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_instance_identity_paths_and_routes_come_from_env(self) -> None:
        output = self._run(
            {
                "INSTANCE_NAME": "Delirya",
                "INSTANCE_SLUG": "delirya",
                "AGENT_NAME": "DELIRYA",
                "AGENT_DISPLAY_NAME": "Delirya",
                "DATA_DIR": "./tmp/delirya-data",
                "STATIC_DIR": "./tmp/delirya-static",
                "PROMPTS_DIR": "./tmp/delirya-prompts",
                "TTS_CACHE_DIR": "./tmp/delirya-tts",
                "SCHEDULER_TASKS_FILE": "./tmp/delirya-tasks.json",
                "MAX_MARKER_FILE": "./tmp/delirya-marker.json",
                "MAX_WEBHOOK_ALIAS_PATH": "",
                "SECRET_HEADER": "X-Delirya-Secret",
                "SERVER_PORT": "8020",
            },
            """
import json
from server import config
from server.sources.max import router
print(json.dumps({
    "name": config.INSTANCE_NAME,
    "slug": config.INSTANCE_SLUG,
    "agent": config.AGENT_NAME,
    "data": str(config.DATA_DIR),
    "static": str(config.STATIC_DIR),
    "prompts": str(config.PROMPTS_DIR),
    "tts": str(config.TTS_CACHE_DIR),
    "tasks": str(config.SCHEDULER_TASKS_FILE),
    "marker": str(config.MAX_MARKER_FILE),
    "webhook": config.MAX_WEBHOOK_PATH,
    "alias": config.MAX_WEBHOOK_ALIAS_PATH,
    "header": config.SECRET_HEADER,
    "port": config.SERVER_PORT,
    "routes": [route.path for route in router.routes],
}, ensure_ascii=False))
""",
        )
        data = json.loads(output)
        root = ROOT.resolve()
        self.assertEqual(data["name"], "Delirya")
        self.assertEqual(data["slug"], "delirya")
        self.assertEqual(data["agent"], "DELIRYA")
        self.assertEqual(data["data"], str(root / "tmp/delirya-data"))
        self.assertEqual(data["static"], str(root / "tmp/delirya-static"))
        self.assertEqual(data["prompts"], str(root / "tmp/delirya-prompts"))
        self.assertEqual(data["tts"], str(root / "tmp/delirya-tts"))
        self.assertEqual(data["tasks"], str(root / "tmp/delirya-tasks.json"))
        self.assertEqual(data["marker"], str(root / "tmp/delirya-marker.json"))
        self.assertEqual(data["webhook"], "/delirya/max-webhook")
        self.assertEqual(data["alias"], "")
        self.assertEqual(data["header"], "X-Delirya-Secret")
        self.assertEqual(data["port"], 8020)
        self.assertEqual(data["routes"], ["/delirya/max-webhook"])

    def test_prompt_placeholders_are_rendered_from_config(self) -> None:
        output = self._run(
            {
                "INSTANCE_NAME": "Delirya",
                "DATA_DIR": "./tmp/prompt-data",
                "TRUSTED_MAX_CHAT_IDS": "111,222",
                "TRUSTED_MAX_USER_IDS": "333",
            },
            """
from server import config
from agent import _render_prompt_text
text = _render_prompt_text(
    config.CORE_PROMPT_PATH.read_text(encoding="utf-8")
)
print("{{" in text)
print(config.DATA_DIR.as_posix() in text)
print("111, 222" in text)
print("333" in text)
""",
        )
        self.assertEqual(output.splitlines(), ["False", "True", "True", "True"])


if __name__ == "__main__":
    unittest.main()

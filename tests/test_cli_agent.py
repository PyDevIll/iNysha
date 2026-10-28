"""Тесты автономного ``cli-agent.py`` (одиночный прогон без памяти и истории).

Модуль загружается через importlib, потому что в имени файла дефис и обычный
``import cli_agent`` невозможен. Сеть не используется: LLM подменяется фейковым
клиентом, инструменты — только read-only (``fs_stat``).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cli_agent():
    spec = importlib.util.spec_from_file_location("cli_agent", ROOT / "cli-agent.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cli_agent = _load_cli_agent()
cli_agent._configure_logging(quiet=True)


class _FakeCompletions:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("fake client ran out of responses")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(responses))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


def _answer(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


class IncludedFilesTest(unittest.TestCase):
    def test_globs_and_literals_keep_argument_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "00_context.md").write_text("ctx", encoding="utf-8")
            (base / "01_facts.json").write_text("{}", encoding="utf-8")
            (base / "02_history.md").write_text("hist", encoding="utf-8")

            files = cli_agent.expand_included(
                [str(base / "0[01]_*"), str(base / "02_history.md")]
            )

            self.assertEqual(
                [p.name for p in files],
                ["00_context.md", "01_facts.json", "02_history.md"],
            )

    def test_missing_glob_and_missing_file_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                cli_agent.expand_included([str(Path(tmp) / "nope*")])
            with self.assertRaises(FileNotFoundError):
                cli_agent.expand_included([str(Path(tmp) / "nope.md")])

    def test_messages_put_files_after_the_run_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Path(tmp) / "00_context.md"
            ctx.write_text("ОПОРНЫЙ КОНТЕКСТ", encoding="utf-8")

            messages = cli_agent.build_messages(
                "Ты трикстер.", "Шаг 02: предыстория.", [ctx]
            )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("трикстер", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        user = messages[1]["content"]
        self.assertLess(user.index("Шаг 02"), user.index("### FILE:"))
        self.assertIn("ОПОРНЫЙ КОНТЕКСТ", user)


class ToolRegistryTest(unittest.TestCase):
    def test_only_the_declared_tools_are_exposed(self) -> None:
        registry = cli_agent.ToolRegistry()
        self.assertEqual(registry.names, sorted(cli_agent.TOOL_NAMES))
        self.assertEqual(
            {schema["function"]["name"] for schema in registry.schemas()},
            set(cli_agent.TOOL_NAMES),
        )

    def test_server_only_tools_are_not_reachable(self) -> None:
        registry = cli_agent.ToolRegistry()
        for name in ("tavily_search", "memory_search", "max_send_message", "start_tool"):
            self.assertNotIn(name, registry.names)

    def test_exec_tools_are_present(self) -> None:
        registry = cli_agent.ToolRegistry()
        self.assertIn("exec_shell", registry.names)
        self.assertIn("exec_python", registry.names)

    def test_encoding_repair_is_available(self) -> None:
        self.assertIn("fs_repair_encoding", cli_agent.ToolRegistry().names)

    def test_bad_json_arguments_become_a_tool_error_not_a_crash(self) -> None:
        registry = cli_agent.ToolRegistry()

        async def _run():
            return await registry.call("fs_stat", "{not json", timeout=5.0)

        content, args = asyncio.run(_run())
        self.assertIsNone(args)
        self.assertIn("invalid JSON", content)


class LoopTest(unittest.TestCase):
    def test_tool_round_trip_then_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.md"
            probe.write_text("привет", encoding="utf-8")

            client = FakeClient(
                [
                    _tool_call("call_1", "fs_read", {"file": str(probe)}),
                    _answer("готово"),
                ]
            )
            agent = cli_agent.CliAgent(
                client,
                model="deepseek-flash",
                max_tokens=1000,
                temperature=cli_agent.DEFAULT_TEMPERATURE,
                reasoning=True,
                max_iterations=5,
                tool_timeout=30.0,
                tool_output_limit=4000,
                registry=cli_agent.ToolRegistry(),
            )
            messages = cli_agent.build_messages("Ты трикстер.", "Что в файле?", [])
            answer = asyncio.run(agent.run(messages))

        self.assertEqual(answer, "готово")
        self.assertEqual(agent.iterations, 2)
        self.assertEqual(agent.tool_calls, 1)

        # второй запрос содержит ровно один tool-результат на объявленный id
        second = client.calls[1]["messages"]
        tool_messages = [m for m in second if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        self.assertIn("привет", tool_messages[0]["content"])

    def test_temperature_and_thinking_are_sent(self) -> None:
        client = FakeClient([_answer("ок")])
        agent = cli_agent.CliAgent(
            client,
            model="deepseek-flash",
            max_tokens=1000,
            temperature=1.3,
            reasoning=True,
            max_iterations=2,
            tool_timeout=30.0,
            tool_output_limit=4000,
        )
        asyncio.run(agent.run([{"role": "user", "content": "привет"}]))

        call = client.calls[0]
        self.assertEqual(call["temperature"], 1.3)
        self.assertEqual(call["extra_body"]["thinking"]["type"], "enabled")
        self.assertIsNone(call["tools"])

    def test_missing_final_answer_hits_iteration_limit(self) -> None:
        loop_forever = [_tool_call(f"c{i}", "fs_stat", {"path": "."}) for i in range(3)]
        agent = cli_agent.CliAgent(
            FakeClient(loop_forever),
            model="deepseek-flash",
            max_tokens=100,
            temperature=1.0,
            reasoning=False,
            max_iterations=2,
            tool_timeout=10.0,
            tool_output_limit=1000,
            registry=cli_agent.ToolRegistry(),
        )
        with self.assertRaises(RuntimeError):
            asyncio.run(agent.run([{"role": "user", "content": "привет"}]))


class OutputTest(unittest.TestCase):
    def test_writes_file_and_creates_parents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "deep" / "nested" / "out.md"
            written = cli_agent.write_output("текст", str(target))
            self.assertEqual(written, target)
            self.assertEqual(target.read_text(encoding="utf-8"), "текст\n")
            self.assertEqual(
                [p.name for p in target.parent.iterdir() if p.name.startswith(".cli-agent-")],
                [],
            )

    def test_dash_means_stdout(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            written = cli_agent.write_output("в stdout", "-")
        self.assertIsNone(written)
        self.assertEqual(buffer.getvalue(), "в stdout\n")


class EndToEndTest(unittest.TestCase):
    def test_amain_builds_prompt_and_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "identity.md").write_text("Ты трикстер.", encoding="utf-8")
            (base / "step.tpl.md").write_text("Напиши предысторию.", encoding="utf-8")
            (base / "00_context.md").write_text("ЧТО УЖЕ СКАЗАНО", encoding="utf-8")
            out = base / "step.md"

            ns = cli_agent.parse_args(
                [
                    "--base_prompt_file", str(base / "identity.md"),
                    "--prompt_file_for_run", str(base / "step.tpl.md"),
                    "--included_files", str(base / "00_*"),
                    "--output_file", str(out),
                ]
            )
            client = FakeClient([_answer("финальный текст")])
            asyncio.run(cli_agent.amain(ns, client=client))

            self.assertEqual(out.read_text(encoding="utf-8"), "финальный текст\n")
            user = client.calls[0]["messages"][1]["content"]
            self.assertIn("ЧТО УЖЕ СКАЗАНО", user)
            self.assertEqual(client.calls[0]["temperature"], 1.3)
            self.assertIsNotNone(client.calls[0]["tools"])


if __name__ == "__main__":
    unittest.main()

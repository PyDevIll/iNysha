"""Voice tool policy, exact-once deferred results and non-blocking speech."""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent
from builtin_tools import additional_tools
from server.context import RequestContext, current_request, is_voice_context
from server.remote_tools import register_all as register_client_tools
from server.routing import VoiceReply
from server.voice import VoiceHub, VoiceSession
from tool_registry import VOICE_DIRECT_TOOLS, ToolRegistry


class _FakeWebSocket:
    def __init__(self) -> None:
        self.json_messages: list[dict] = []
        self.binary_messages: list[bytes] = []

    async def send_json(self, payload: dict) -> None:
        self.json_messages.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.binary_messages.append(bytes(data))


class _FakeFunction:
    def __init__(self, name: str, arguments: str = "{}") -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name: str, arguments: str = "{}") -> None:
        self.id = "call-1"
        self.function = _FakeFunction(name, arguments)


def _context(source: str, session_id: str | None) -> RequestContext:
    return RequestContext(
        request_id="test",
        source=source,
        client_id="desktop",
        voice_session_id=session_id,
    )


class VoiceToolPolicyTest(unittest.TestCase):
    def test_strict_voice_allowlist_and_full_text_set(self) -> None:
        registry = ToolRegistry()
        registry.hot_reload()
        register_client_tools(registry)

        text = {
            tool["function"]["name"]
            for tool in registry.get_openai_tools(voice_mode=False)
        }
        voice = {
            tool["function"]["name"]
            for tool in registry.get_openai_tools(voice_mode=True)
        }

        self.assertNotIn("start_tool", text)
        self.assertNotIn("check_tool", text)
        self.assertIn("start_tool", voice)
        self.assertIn("check_tool", voice)

        for name in (
            "exec_python",
            "exec_shell",
            "tts_generate",
            "tts_speak_aloud",
            "client_record_audio",
            "get_screenshot",
            "tavily_search",
            "vision_analyze",
            "max_send_message",
        ):
            self.assertIn(name, text)
            self.assertNotIn(name, voice)

        for name in ("fs_read", "fs_stat", "list_clients", "schedule_task"):
            self.assertIn(name, voice)
        self.assertLessEqual(VOICE_DIRECT_TOOLS, voice)

    def test_voice_context_uses_session_id(self) -> None:
        cases = (
            ("voice", "s1", True),
            ("tool_result", "s1", True),
            ("client_text", None, False),
            ("max", None, False),
            ("api", "s1", True),  # only the server can set this field
        )
        for source, session_id, expected in cases:
            token = current_request.set(_context(source, session_id))
            try:
                self.assertIs(is_voice_context(), expected, source)
            finally:
                current_request.reset(token)


class DirectToolPolicyTest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_rejects_blocking_tool_in_voice(self) -> None:
        agent = object.__new__(Agent)
        token = current_request.set(_context("voice", "s1"))
        try:
            result = await agent._execute_single_tool(
                _FakeToolCall("exec_shell", '{"command": "echo should-not-run"}')
            )
        finally:
            current_request.reset(token)

        self.assertIn("not available for direct calls", result["content"])
        self.assertNotIn("should-not-run", result["content"])

    async def test_agent_rejects_voice_only_helper_in_text(self) -> None:
        agent = object.__new__(Agent)
        token = current_request.set(_context("client_text", None))
        try:
            result = await agent._execute_single_tool(
                _FakeToolCall("start_tool", '{"tool": "exec_shell"}')
            )
        finally:
            current_request.reset(token)

        self.assertIn("only during voice turns", result["content"])

    async def test_start_tool_gate_accepts_voice_and_tool_result(self) -> None:
        for source, session_id, expected in (
            ("voice", "s1", "not found"),
            ("tool_result", "s1", "not found"),
            ("client_text", None, "only in a voice turn"),
        ):
            token = current_request.set(_context(source, session_id))
            try:
                result = await additional_tools.start_tool("definitely-missing")
            finally:
                current_request.reset(token)
            self.assertIn(expected, result, source)


class DeferredDeliveryTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        additional_tools._TOOL_TASKS.clear()

    def _task(self, session_id: str = "s1") -> dict:
        return {
            "tool": "tavily_search",
            "status": "ready",
            "result": "secret result",
            "error": None,
            "started_at": time.time(),
            "finished_at": time.time(),
            "voice_session_id": session_id,
            "client_id": "desktop",
            "chat_id": None,
            "auto_deliver": True,
            "delivered": False,
            "consumed": False,
        }

    async def test_check_tool_claims_result_once_and_is_session_scoped(
        self,
    ) -> None:
        task = self._task("s1")
        additional_tools._TOOL_TASKS["t1"] = task

        token = current_request.set(_context("voice", "s1"))
        try:
            first = await additional_tools.check_tool("t1")
            second = await additional_tools.check_tool("t1")
            listed = await additional_tools.list_tool_tasks()
        finally:
            current_request.reset(token)

        self.assertIn("secret result", first)
        self.assertTrue(task["consumed"])
        self.assertFalse(task["auto_deliver"])
        self.assertNotIn("secret result", second)
        self.assertIn("already claimed", second)
        self.assertIn("t1", listed)

        token = current_request.set(_context("voice", "s2"))
        try:
            other = await additional_tools.check_tool("t1")
            other_list = await additional_tools.list_tool_tasks()
        finally:
            current_request.reset(token)

        self.assertIn("not found", other)
        self.assertNotIn("t1", other_list)

    async def test_consumed_task_is_not_auto_delivered(self) -> None:
        task = self._task("s1")
        task["consumed"] = True
        task["auto_deliver"] = False
        additional_tools._TOOL_TASKS["t1"] = task

        # Must return before touching the real runtime/queue.
        await additional_tools._deliver_tool_result(task, "t1")
        self.assertFalse(task["delivered"])

    async def test_auto_delivery_enqueues_exactly_once(self) -> None:
        from unittest.mock import patch

        class FakeQueue:
            def __init__(self) -> None:
                self.requests: list = []

            async def put(self, request) -> None:
                self.requests.append(request)

            def qsize(self) -> int:
                return len(self.requests)

        class FakeRuntime:
            def __init__(self) -> None:
                self.queue = FakeQueue()
                self.bot = None

        class FakeHub:
            def is_live(self, client_id=None, *, session_id=None) -> bool:
                return session_id == "s1"

        task = self._task("s1")
        fake_rt = FakeRuntime()
        with patch("server.runtime.runtime", fake_rt), patch(
            "server.voice.voice_hub", FakeHub()
        ):
            await additional_tools._deliver_tool_result(task, "t1")
            await additional_tools._deliver_tool_result(task, "t1")

        self.assertTrue(task["delivered"])
        self.assertEqual(len(fake_rt.queue.requests), 1)
        request = fake_rt.queue.requests[0]
        self.assertEqual(request.source, "tool_result")
        self.assertEqual(request.voice_session_id, "s1")
        self.assertEqual(request.tool_task_id, "t1")


class NonBlockingSpeechTest(unittest.IsolatedAsyncioTestCase):
    async def test_voice_reply_prefers_async_hub(self) -> None:
        class Hub:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            async def send_spoken_async(
                self, client_id: str, text: str, *, session_id=None
            ) -> bool:
                self.calls.append((client_id, text, session_id))
                return True

        hub = Hub()
        reply = VoiceReply(hub, "desktop", session_id="s1")
        await reply.send_text("hello")
        self.assertEqual(hub.calls, [("desktop", "hello", "s1")])

    async def test_scheduled_speech_waits_for_foreground_turn(self) -> None:
        websocket = _FakeWebSocket()

        async def fake_tts(text: str, backend: str, *, on_status=None):
            yield b"pcm"

        session = VoiceSession(
            websocket,
            "desktop",
            "Desktop",
            hub=VoiceHub(),
            session_id="s1",
            tts_stream=fake_tts,
        )
        session.session_open = True
        session._turn_idle.clear()
        task = session.schedule_speech("later")
        self.assertIsNotNone(task)

        await asyncio.sleep(0)
        self.assertEqual(websocket.binary_messages, [])

        session._turn_idle.set()
        await task
        self.assertEqual(websocket.binary_messages, [b"pcm"])
        self.assertTrue(
            any(
                message.get("type") == "done"
                for message in websocket.json_messages
            )
        )

    async def test_hub_schedules_without_waiting(self) -> None:
        class Session:
            session_id = "s1"
            client_id = "desktop"
            connected_at = 1.0

            def __init__(self) -> None:
                self.scheduled: list[tuple[str, str]] = []

            def schedule_speech(self, text: str, *, source="background"):
                self.scheduled.append((text, source))
                return object()

        hub = VoiceHub()
        session = Session()
        hub._sessions["s1"] = session
        hub._by_client["desktop"] = {"s1"}

        ok = await hub.send_spoken_async("desktop", "hello", session_id="s1")
        self.assertTrue(ok)
        self.assertEqual(session.scheduled, [("hello", "background")])


if __name__ == "__main__":
    unittest.main()

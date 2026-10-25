"""Тесты голосовых сессий: маршрутизация, сериализация TTS и lifecycle."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.routing import VoiceReply
from server.voice import VoiceHub, VoiceSession, stream_tts


class FakeWebSocket:
    def __init__(self) -> None:
        self.json_messages: list[dict] = []
        self.binary_messages: list[bytes] = []
        self.incoming: asyncio.Queue[dict] = asyncio.Queue()

    async def send_json(self, payload: dict) -> None:
        self.json_messages.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.binary_messages.append(bytes(data))

    async def receive(self) -> dict:
        return await self.incoming.get()


class FakeQueue:
    def __init__(self) -> None:
        self.requests: list = []

    async def put(self, request) -> None:
        self.requests.append(request)
        await request.reply.send_text("answer")

    def qsize(self) -> int:
        return 0


class FakeVoiceSession:
    def __init__(
        self, session_id: str, client_id: str, connected_at: float
    ) -> None:
        self.session_id = session_id
        self.client_id = client_id
        self.connected_at = connected_at
        self.spoken: list[tuple[str, str]] = []
        self.waited: list[float] = []

    async def speak(self, text: str, *, source: str = "background") -> bool:
        self.spoken.append((text, source))
        return True

    async def wait_idle(self, timeout: float = 30.0) -> None:
        self.waited.append(timeout)


class TTSStreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_google_backend_delegates_to_google_helper(self) -> None:
        async def fake_google(text: str, on_status=None):
            yield b"google"

        with patch("server.voice.stream_google_pcm", fake_google):
            chunks = [chunk async for chunk in stream_tts("hi", "google")]

        self.assertEqual(chunks, [b"google"])

    async def test_eleven_backend_delegates_to_eleven_helper(self) -> None:
        async def fake_eleven(text: str, **kwargs):
            yield b"eleven"

        with patch("server.voice.stream_pcm", fake_eleven):
            chunks = [chunk async for chunk in stream_tts("hi", "eleven")]

        self.assertEqual(chunks, [b"eleven"])


class VoiceHubTest(unittest.IsolatedAsyncioTestCase):
    async def test_routes_by_session_id_when_client_has_multiple_sessions(
        self,
    ) -> None:
        hub = VoiceHub()
        first = FakeVoiceSession("s1", "desktop", 1.0)
        second = FakeVoiceSession("s2", "desktop", 2.0)
        hub.register(first)
        hub.register(second)

        self.assertTrue(
            await hub.send_spoken(
                "desktop", "hello", session_id="s1"
            )
        )
        self.assertEqual(first.spoken, [("hello", "background")])
        self.assertEqual(second.spoken, [])

        await hub.wait_idle("desktop", session_id="s2", timeout=5.0)
        self.assertEqual(second.waited, [5.0])
        self.assertEqual(first.waited, [])

    async def test_old_session_cleanup_does_not_remove_new_session(self) -> None:
        hub = VoiceHub()
        old = FakeVoiceSession("old", "desktop", 1.0)
        new = FakeVoiceSession("new", "desktop", 2.0)
        hub.register(old)
        hub.register(new)

        hub.unregister(old)

        self.assertIsNone(hub.get("old"))
        self.assertIs(hub.get("new"), new)
        self.assertEqual(hub.live_clients(), ["desktop"])

    async def test_legacy_client_fallback_routes_newest(self) -> None:
        hub = VoiceHub()
        first = FakeVoiceSession("s1", "desktop", 1.0)
        second = FakeVoiceSession("s2", "desktop", 2.0)
        hub.register(first)
        hub.register(second)

        await hub.send_spoken("desktop", "fallback")

        self.assertEqual(first.spoken, [])
        self.assertEqual(second.spoken, [("fallback", "background")])


class VoiceSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_speak_streams_chunks_and_reports_session_ready(self) -> None:
        websocket = FakeWebSocket()

        async def fake_tts(text: str, backend: str, *, on_status=None):
            if on_status is not None:
                on_status(
                    {
                        "index": 0,
                        "total": 1,
                        "text": text,
                        "mp3_bytes": 10,
                        "pcm_bytes": 4,
                        "elapsed_s": 0.1,
                    }
                )
            yield b"aa"
            yield b"bb"

        session = VoiceSession(
            websocket,
            "desktop",
            "Desktop",
            hub=VoiceHub(),
            session_id="s1",
            queue=FakeQueue(),
            tts_stream=fake_tts,
        )

        self.assertTrue(await session.speak("Hello"))
        self.assertEqual(websocket.binary_messages, [b"aa", b"bb"])
        self.assertIn(
            {"type": "status", "status": "tts"},
            websocket.json_messages,
        )

    async def test_speak_serializes_foreground_and_background(self) -> None:
        websocket = FakeWebSocket()
        active = 0
        max_active = 0

        async def slow_tts(text: str, backend: str, *, on_status=None):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.01)
                yield b"x"
                await asyncio.sleep(0.01)
                yield b"y"
            finally:
                active -= 1

        session = VoiceSession(
            websocket,
            "desktop",
            "Desktop",
            hub=VoiceHub(),
            session_id="s1",
            queue=FakeQueue(),
            tts_stream=slow_tts,
        )

        await asyncio.gather(
            session.speak("first"),
            session.speak("second"),
        )

        self.assertEqual(max_active, 1)

    async def test_ready_contains_session_id_and_unregisters_on_disconnect(
        self,
    ) -> None:
        websocket = FakeWebSocket()
        websocket.incoming.put_nowait({"type": "websocket.disconnect"})
        hub = VoiceHub()
        session = VoiceSession(
            websocket,
            "desktop",
            "Desktop",
            hub=hub,
            session_id="s1",
        )

        await session.run()

        self.assertEqual(
            websocket.json_messages[0],
            {
                "type": "ready",
                "client_id": "desktop",
                "session_id": "s1",
                "name": "Desktop",
            },
        )
        self.assertIsNone(hub.get("s1"))

    async def test_turn_request_carries_exact_session_id(self) -> None:
        websocket = FakeWebSocket()
        queue = FakeQueue()

        async def fake_transcribe(pcm: bytes, sample_rate: int) -> str:
            return "hello"

        async def fake_tts(text: str, backend: str, *, on_status=None):
            yield b"pcm"

        session = VoiceSession(
            websocket,
            "desktop",
            "Desktop",
            hub=VoiceHub(),
            session_id="s1",
            queue=queue,
            transcribe=fake_transcribe,
            tts_stream=fake_tts,
        )

        await session._run_turn(b"\x00\x10" * 8000)

        self.assertEqual(len(queue.requests), 1)
        request = queue.requests[0]
        self.assertEqual(request.client_id, "desktop")
        self.assertEqual(request.voice_session_id, "s1")
        self.assertEqual(request.source, "voice")
        self.assertIn({"type": "final", "text": "answer"}, websocket.json_messages)
        self.assertIn(b"pcm", websocket.binary_messages)


class VoiceReplyTest(unittest.IsolatedAsyncioTestCase):
    async def test_reply_targets_session_id(self) -> None:
        class Hub:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            async def send_spoken(
                self, client_id: str, text: str, *, session_id=None
            ) -> bool:
                self.calls.append((client_id, text, session_id))
                return True

        hub = Hub()
        reply = VoiceReply(hub, "desktop", session_id="s1")
        await reply.send_text("hello")

        self.assertEqual(
            hub.calls,
            [("desktop", "hello", "s1")],
        )


if __name__ == "__main__":
    unittest.main()

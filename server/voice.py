"""Low-latency voice WebSocket for desktop and browser clients.

The protocol mirrors the working max-voice-assistant mini-app:

    client -> server: {"action": "listen_start" | "stop" | "playback_done"}
    client -> server: raw 16 kHz / 16-bit / mono PCM frames
    server -> client: {"type": "ready" | "vad" | "status" | "stt" |
                              "final" | "error" | "done"}
    server -> client: raw 16 kHz / 16-bit / mono PCM (Google TTS / ElevenLabs)

VAD and STT run on the server, so the client stays a thin microphone/speaker
body.  The shared agent queue is used for the reasoning turn, keeping one
serialized context across MAX, API and desktop voice clients.

Each WebSocket gets its own ``VoiceSession``.  ``VoiceHub`` keeps sessions by
``session_id`` (not only by ``client_id``), so two tabs/clients with the same
desktop id cannot overwrite each other; replies are routed to the session that
originated the turn.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from lib.elevenlabs_tts import stream_pcm
from server import config
from server.audio import (
    MIN_AUDIO_BYTES,
    SILENCE_RMS_THRESHOLD,
    is_likely_hallucination,
    pcm_rms,
    sanitize_for_tts,
    transcribe_pcm,
)
from server.queue import Request
from server.routing import FutureReply
from server.runtime import runtime
from server.tts_google import stream_google_pcm
from server.vad import VadSegmenter

router = APIRouter(tags=["voice"])

FRAME_SECONDS = 0.03

TTSStatusCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


def _normalize_tts_backend(backend: str) -> str:
    return "google" if backend == "google" else "eleven"


async def stream_tts(
    text: str,
    backend: str = "google",
    *,
    on_status: TTSStatusCallback | None = None,
) -> AsyncIterator[bytes]:
    """Yield raw 16 kHz PCM for ``text``.

    This helper owns backend selection only: no WebSocket, no session state,
    no locks.  Google streams short pieces with prefetch; ElevenLabs streams
    one long response.
    """
    backend = _normalize_tts_backend(backend)
    if backend == "google":
        async for chunk in stream_google_pcm(text, on_status=on_status):
            yield chunk
        return

    async for chunk in stream_pcm(
        text,
        output_format=config.VOICE_TTS_FORMAT,
        optimize_streaming_latency=config.VOICE_TTS_LATENCY,
    ):
        yield chunk


class VoiceHub:
    """Registry of live voice sessions and the routing facade for replies.

    Sessions are keyed by ``session_id``; ``client_id`` is only an index.
    That distinction is what makes multiple tabs/desktop clients safe: an old
    session's cleanup cannot unregister a newer session with the same client
    id, and a reply can target the exact WebSocket that originated the turn.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}
        self._by_client: dict[str, set[str]] = {}

    def register(self, session: Any) -> None:
        self._sessions[session.session_id] = session
        self._by_client.setdefault(session.client_id, set()).add(
            session.session_id
        )
        logger.info(
            f"voice hub: session {session.session_id} "
            f"({session.client_id}) registered"
        )

    def unregister(self, session: Any) -> None:
        if self._sessions.get(session.session_id) is not session:
            logger.debug(
                f"voice hub: stale session {session.session_id} cleanup ignored"
            )
            return
        self._sessions.pop(session.session_id, None)
        ids = self._by_client.get(session.client_id)
        if ids is not None:
            ids.discard(session.session_id)
            if not ids:
                self._by_client.pop(session.client_id, None)
        logger.info(
            f"voice hub: session {session.session_id} "
            f"({session.client_id}) closed"
        )

    def get(self, session_id: str) -> Any | None:
        return self._sessions.get(session_id)

    def for_client(self, client_id: str) -> list[Any]:
        return [
            self._sessions[sid]
            for sid in self._by_client.get(client_id, set())
            if sid in self._sessions
        ]

    def is_live(
        self, client_id: str | None = None, *, session_id: str | None = None
    ) -> bool:
        if session_id is not None:
            return session_id in self._sessions
        if client_id is not None:
            return bool(self.for_client(client_id))
        return bool(self._sessions)

    def live_clients(self) -> list[str]:
        return sorted(
            client_id
            for client_id in self._by_client
            if self.for_client(client_id)
        )

    def resolve(
        self,
        client_id: str | None = None,
        *,
        session_id: str | None = None,
    ) -> Any | None:
        """Find a session by exact id, then by client id.

        ``client_id`` is a legacy fallback used by older tool-result tasks.
        New turns carry ``session_id``, so the fallback is only exercised when
        the exact session is unavailable.
        """
        if session_id is not None:
            return self._sessions.get(session_id)
        if client_id is not None:
            sessions = self.for_client(client_id)
            if not sessions:
                return None
            if len(sessions) > 1:
                sessions.sort(key=lambda item: item.connected_at)
                logger.warning(
                    f"voice hub: {len(sessions)} sessions for client "
                    f"{client_id!r}; routing to newest "
                    f"{sessions[-1].session_id}"
                )
                return sessions[-1]
            return sessions[0]
        if len(self._sessions) == 1:
            return next(iter(self._sessions.values()))
        return None

    async def send_spoken(
        self,
        client_id: str | None = None,
        text: str = "",
        *,
        session_id: str | None = None,
    ) -> bool:
        """Pronounce text in a live session. Empty text means silence."""
        session = self.resolve(client_id, session_id=session_id)
        target = session_id or client_id or "<any>"
        if session is None:
            logger.info(f"voice hub: no live session for {target}, mute")
            return False
        if not (text or "").strip():
            logger.info(
                f"voice hub: session {session.session_id} stays silent "
                "(empty answer)"
            )
            return False
        started = time.time()
        ok = await session.speak(text, source="background")
        logger.info(
            f"voice hub: session {session.session_id} spoken, "
            f"{time.time() - started:.1f}s playback"
        )
        return ok

    async def wait_idle(
        self,
        client_id: str | None = None,
        timeout: float = 30.0,
        *,
        session_id: str | None = None,
    ) -> None:
        """Wait for the current utterance (maximum ``timeout`` seconds)."""
        session = self.resolve(client_id, session_id=session_id)
        if session is None:
            return
        await session.wait_idle(timeout)


voice_hub = VoiceHub()


async def _authenticate(websocket: WebSocket) -> dict[str, Any] | None:
    if not config.CLIENT_TOKEN:
        await websocket.close(code=1011, reason="CLIENT_TOKEN is not configured")
        return None
    try:
        hello = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except Exception:
        await websocket.close(code=4001, reason="Expected hello message")
        return None
    if hello.get("type") != "hello":
        await websocket.close(code=4001, reason="Expected hello message")
        return None
    token = str(hello.get("token") or "")
    if not hmac.compare_digest(token, config.CLIENT_TOKEN):
        logger.warning("voice client authentication failed")
        await websocket.close(code=4001, reason="Authentication failed")
        return None
    return hello


class VoiceSession:
    """One live voice WebSocket: protocol loop, VAD, turns and TTS."""

    def __init__(
        self,
        websocket: WebSocket,
        client_id: str,
        name: str,
        *,
        hub: VoiceHub,
        tts_backend: str = "google",
        trusted: bool | None = None,
        session_id: str | None = None,
        queue: Any | None = None,
        transcribe: Callable[[bytes, int], Awaitable[str]] | None = None,
        tts_stream: Callable[..., AsyncIterator[bytes]] | None = None,
    ) -> None:
        self.websocket = websocket
        self.client_id = client_id
        self.name = name
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.hub = hub
        self.tts_backend = _normalize_tts_backend(tts_backend)
        self.trusted = config.CLIENT_TRUSTED if trusted is None else trusted
        self.queue = queue if queue is not None else runtime.queue
        self._transcribe = transcribe or transcribe_pcm
        self._tts_stream = tts_stream or stream_tts
        self.connected_at = time.time()

        self.send_lock = asyncio.Lock()
        # Serializes whole utterances (not just individual chunks). Foreground
        # and background speech both go through ``speak``, so they cannot
        # interleave PCM in one socket.
        self.playback_lock = asyncio.Lock()
        self.segmenter = VadSegmenter(
            aggressiveness=2,
            preroll_ms=400,
            min_speech_ms=250,
            hangover_ms=700,
            max_speech_ms=config.VOICE_MAX_SECONDS * 1000,
        )
        self.turn_task: asyncio.Task | None = None
        self.session_open = False
        self.accept_audio = False
        self._closed = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.websocket.send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        async with self.send_lock:
            await self.websocket.send_bytes(data)

    async def run(self) -> None:
        logger.info(
            f"voice session {self.session_id} connected: "
            f"{self.client_id!r} ({self.name})"
        )
        self.hub.register(self)
        try:
            await self.send_json(
                {
                    "type": "ready",
                    "client_id": self.client_id,
                    "session_id": self.session_id,
                    "name": self.name,
                }
            )
            await self._receive_loop()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception(
                f"voice socket failed for session {self.session_id}"
            )
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.session_open = False
        self.accept_audio = False
        self.hub.unregister(self)
        await self._cancel_turn()
        logger.info(
            f"voice session {self.session_id} disconnected: "
            f"{self.client_id!r} ({self.name})"
        )

    async def _cancel_turn(self) -> None:
        task = self.turn_task
        if task is None or task.done():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _receive_loop(self) -> None:
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue

            if "bytes" in message and message["bytes"] is not None:
                if not self.accept_audio:
                    continue
                for event in self.segmenter.feed(message["bytes"]):
                    if event["type"] == "speech_start":
                        await self.send_json(
                            {"type": "vad", "event": "speech_start"}
                        )
                    elif event["type"] == "speech_end":
                        self.accept_audio = False
                        await self.send_json(
                            {"type": "vad", "event": "speech_end"}
                        )
                        if self.turn_task is None or self.turn_task.done():
                            self.turn_task = asyncio.create_task(
                                self._run_turn(event["audio"]),
                                name=f"voice-turn-{self.session_id}",
                            )
                continue

            if "text" not in message or message["text"] is None:
                continue

            try:
                data = json.loads(message["text"])
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            await self._handle_control(data)

    async def _handle_control(self, data: dict[str, Any]) -> None:
        action = data.get("action")

        # Браузерная сборка MAX WebApp может прислать initData. В MAX
        # браузерного клиента не пускаем, поэтому сообщение игнорируем.
        if data.get("type") == "auth":
            logger.debug(f"voice {self.session_id}: auth message ignored")
            return

        if action == "listen_start":
            self.session_open = True
            await self._cancel_turn()
            self.segmenter.reset()
            self.accept_audio = True
            logger.debug(f"voice {self.session_id}: listening")

        elif action == "stop":
            self.session_open = False
            self.accept_audio = False
            self.segmenter.reset()
            await self._cancel_turn()
            logger.debug(f"voice {self.session_id}: stopped")

        elif action == "playback_done":
            if self.session_open and (
                self.turn_task is None or self.turn_task.done()
            ):
                self.segmenter.reset()
                self.accept_audio = True
                logger.debug(
                    f"voice {self.session_id}: playback done, listening"
                )

    async def _run_turn(self, pcm: bytes) -> None:
        """STT -> shared agent queue -> streaming TTS."""
        started = time.time()
        try:
            if (
                len(pcm) < MIN_AUDIO_BYTES
                or pcm_rms(pcm) < SILENCE_RMS_THRESHOLD
            ):
                await self.send_json({"type": "status", "status": "no_speech"})
                await self.send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            await self.send_json({"type": "status", "status": "stt"})
            try:
                transcript = await self._transcribe(
                    pcm, config.VOICE_SAMPLE_RATE
                )
            except Exception as exc:
                logger.exception("voice STT failed")
                await self.send_json(
                    {"type": "error", "message": f"STT error: {exc}"}
                )
                await self.send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            if not transcript or is_likely_hallucination(transcript):
                logger.info(
                    f"voice {self.session_id}: no usable transcript"
                )
                await self.send_json({"type": "status", "status": "no_speech"})
                await self.send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            logger.info(
                f"voice {self.session_id}: transcript "
                f"({len(transcript)} chars) {transcript[:120]!r}"
            )
            await self.send_json({"type": "stt", "text": transcript})

            loop = asyncio.get_running_loop()
            future: asyncio.Future[str] = loop.create_future()
            header = (
                f"[Voice client: {self.name} | client_id={self.client_id} | "
                f"trusted={'true' if self.trusted else 'false'}]"
            )
            request = Request(
                text=f"{header}\n{transcript}",
                source="voice",
                client_id=self.client_id,
                voice_session_id=self.session_id,
                trusted=self.trusted,
                reply=FutureReply(future),
            )
            await self.queue.put(request)
            logger.info(
                f"[{request.id}] queued voice turn from session "
                f"{self.session_id} (queue left={self.queue.qsize()})"
            )

            try:
                answer = await asyncio.wait_for(
                    future, timeout=config.VOICE_TURN_TIMEOUT
                )
            except asyncio.TimeoutError:
                await self.send_json(
                    {"type": "error", "message": "agent turn timed out"}
                )
                await self.send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return
            except Exception as exc:
                logger.exception(
                    f"voice agent turn failed for {self.session_id}"
                )
                await self.send_json({"type": "error", "message": str(exc)})
                await self.send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            answer = (answer or "").strip()
            await self.send_json({"type": "final", "text": answer})
            await self.speak(answer, source="turn")
            await self.send_json(
                {"type": "done", "elapsed": time.time() - started}
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"voice turn failed for {self.session_id}")
            with suppress(Exception):
                await self.send_json(
                    {"type": "error", "message": "voice pipeline failed"}
                )
                await self.send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )

    async def speak(self, text: str, *, source: str = "background") -> bool:
        """Synthesize and stream one utterance to this session.

        The playback lock is held for the whole utterance, so foreground and
        background turns cannot interleave.  It covers socket sending, not
        real speaker playback: the browser's ``playback_done`` remains an
        advisory signal for microphone re-enable.
        """
        speech = sanitize_for_tts(text or "")
        if not speech:
            return False

        async with self.playback_lock:
            await self.send_json({"type": "status", "status": "tts"})
            started = time.time()
            bytes_sent = 0
            pieces_seen = 0

            def on_tts_status(info: dict[str, Any]) -> None:
                """Log each ready Google TTS piece."""
                nonlocal pieces_seen
                pieces_seen = int(info["total"])
                logger.info(
                    f"voice {self.session_id}: google TTS piece "
                    f"{info['index'] + 1}/{info['total']} "
                    f"({len(info['text'])} chars, mp3={info['mp3_bytes']}B, "
                    f"pcm={info['pcm_bytes']}B, {info['elapsed_s']}s)"
                )

            try:
                async for chunk in self._tts_stream(
                    speech,
                    self.tts_backend,
                    on_status=on_tts_status,
                ):
                    bytes_sent += len(chunk)
                    await self.send_bytes(chunk)
            except Exception as exc:
                logger.exception(
                    f"voice TTS streaming failed "
                    f"(backend={self.tts_backend}) for "
                    f"{self.session_id}: {exc}"
                )
                with suppress(Exception):
                    await self.send_json(
                        {"type": "error", "message": "voice TTS failed"}
                    )
                return False

            logger.info(
                f"voice {self.session_id}: {source} answer "
                f"{len(text or '')} chars, TTS backend={self.tts_backend}, "
                f"pieces={pieces_seen if self.tts_backend == 'google' else 1}, "
                f"{bytes_sent} bytes PCM, {time.time() - started:.1f}s"
            )
            return True

    async def wait_idle(self, timeout: float = 30.0) -> None:
        """Wait until this session is not sending a TTS utterance."""
        if not self.playback_lock.locked():
            return
        logger.info(
            f"voice hub: session {self.session_id} is speaking, "
            "background turn waits"
        )
        try:
            async with asyncio.timeout(timeout):
                async with self.playback_lock:
                    pass
        except (asyncio.TimeoutError, TimeoutError):
            logger.info(
                f"voice hub: session {self.session_id} still busy "
                f"after {timeout}s"
            )


async def _serve_voice(
    websocket: WebSocket,
    client_id: str,
    name: str,
    tts_backend: str = "eleven",
    *,
    trusted: bool | None = None,
) -> None:
    """Create one session and run its protocol loop.

    Used by both desktop (``/ws/voice``) and browser
    (``/ws/web-voice``, ``/ws/voice?dev=1``) transports, so the pipeline and
    multi-session routing are shared.
    """
    session = VoiceSession(
        websocket,
        client_id,
        name,
        hub=voice_hub,
        tts_backend=tts_backend,
        trusted=trusted,
    )
    await session.run()


def _web_client_id(websocket: WebSocket) -> str:
    """Short browser client id: ?client=... or random."""
    requested = (websocket.query_params.get("client") or "").strip()
    safe = "".join(ch for ch in requested if ch.isalnum() or ch in "-_")[:32]
    return f"web:{safe or uuid.uuid4().hex[:8]}"


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    """Desktop clients: mandatory HELLO with CLIENT_TOKEN."""
    await websocket.accept()

    # The /inysha/voice/ Vite page uses this route with ?dev=1 and cannot
    # send HELLO, so it runs in browser mode without a token. ?dev=1 keeps
    # the ElevenLabs comparison backend; otherwise Google TTS is used.
    if websocket.query_params.get("dev") == "1":
        await _serve_voice(
            websocket,
            _web_client_id(websocket),
            "web",
            tts_backend="eleven",
        )
        return

    hello = await _authenticate(websocket)
    if hello is None:
        return

    client_id = str(hello.get("client_id") or "voice-client")
    name = str(hello.get("name") or client_id)
    await _serve_voice(websocket, client_id, name, tts_backend="google")


@router.websocket("/ws/web-voice")
async def web_voice_ws(websocket: WebSocket) -> None:
    """Browser variant: no HELLO and no CLIENT_TOKEN (MAX is blocked)."""
    await websocket.accept()
    await _serve_voice(
        websocket, _web_client_id(websocket), "web", tts_backend="google"
    )

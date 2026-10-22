"""Low-latency voice WebSocket for desktop clients.

The protocol mirrors the working max-voice-assistant mini-app:

    client -> server: {"action": "listen_start" | "stop" | "playback_done"}
    client -> server: raw 16 kHz / 16-bit / mono PCM frames
    server -> client: {"type": "ready" | "vad" | "status" | "stt" |
                              "final" | "error" | "done"}
    server -> client: raw 16 kHz / 16-bit / mono PCM (Google TTS / ElevenLabs)

VAD and STT run on the server, so the client stays a thin microphone/speaker
body.  The shared agent queue is used for the reasoning turn, keeping one
serialized context across MAX, API and desktop voice clients.
"""

from __future__ import annotations

import asyncio
import hmac
import time
import uuid
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


async def _serve_voice(
    websocket: WebSocket,
    client_id: str,
    name: str,
    tts_backend: str = "eleven",
) -> None:
    """Общий цикл голосовой сессии: VAD -> STT -> очередь агента -> TTS.

    Используется и desktop-роутом (/ws/voice), и браузерным (/ws/web-voice),
    чтобы протокол и пайплайн не дублировались.

    ``tts_backend``: "google" — бесплатный Google Translate TTS (прод),
    "eleven" — прежний ElevenLabs (страница с ?dev=1 для сравнения).
    """
    tts_backend = "google" if tts_backend == "google" else "eleven"
    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_bytes(data: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(data)

    logger.info(f"voice client connected: {client_id!r} ({name})")
    await send_json({"type": "ready", "client_id": client_id, "name": name})

    segmenter = VadSegmenter(
        aggressiveness=2,
        preroll_ms=400,
        min_speech_ms=250,
        hangover_ms=700,
        max_speech_ms=config.VOICE_MAX_SECONDS * 1000,
    )
    session_open = False
    accept_audio = False
    turn_task: asyncio.Task | None = None

    async def run_turn(pcm: bytes) -> None:
        """STT -> shared agent queue -> streaming TTS."""
        started = time.time()
        try:
            if (
                len(pcm) < MIN_AUDIO_BYTES
                or pcm_rms(pcm) < SILENCE_RMS_THRESHOLD
            ):
                await send_json({"type": "status", "status": "no_speech"})
                await send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            await send_json({"type": "status", "status": "stt"})
            try:
                transcript = await transcribe_pcm(pcm, config.VOICE_SAMPLE_RATE)
            except Exception as exc:
                logger.exception("voice STT failed")
                await send_json(
                    {"type": "error", "message": f"STT error: {exc}"}
                )
                await send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            if not transcript or is_likely_hallucination(transcript):
                logger.info(f"voice {client_id}: no usable transcript")
                await send_json({"type": "status", "status": "no_speech"})
                await send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            logger.info(
                f"voice {client_id}: transcript ({len(transcript)} chars) "
                f"{transcript[:120]!r}"
            )
            await send_json({"type": "stt", "text": transcript})

            loop = asyncio.get_running_loop()
            future: asyncio.Future[str] = loop.create_future()
            header = (
                f"[Voice client: {name} | client_id={client_id} | "
                f"trusted={'true' if config.CLIENT_TRUSTED else 'false'}]"
            )
            request = Request(
                text=f"{header}\n{transcript}",
                source="voice",
                client_id=client_id,
                trusted=config.CLIENT_TRUSTED,
                reply=FutureReply(future),
            )
            await runtime.queue.put(request)
            logger.info(
                f"[{request.id}] queued voice turn from client {client_id} "
                f"(queue left={runtime.queue.qsize()})"
            )

            try:
                answer = await asyncio.wait_for(
                    future, timeout=config.VOICE_TURN_TIMEOUT
                )
            except asyncio.TimeoutError:
                await send_json(
                    {"type": "error", "message": "agent turn timed out"}
                )
                await send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return
            except Exception as exc:
                logger.exception(f"voice agent turn failed for {client_id}")
                await send_json({"type": "error", "message": str(exc)})
                await send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            answer = (answer or "").strip()
            await send_json({"type": "final", "text": answer})
            speech = sanitize_for_tts(answer)
            if not speech:
                await send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )
                return

            await send_json({"type": "status", "status": "tts"})
            bytes_sent = 0
            pieces_seen = 0

            def on_tts_status(info: dict[str, Any]) -> None:
                """Логируем каждый готовый кусок Google TTS."""
                nonlocal pieces_seen
                pieces_seen = int(info["total"])
                logger.info(
                    f"voice {client_id}: google TTS piece "
                    f"{info['index'] + 1}/{info['total']} "
                    f"({len(info['text'])} chars, mp3={info['mp3_bytes']}B, "
                    f"pcm={info['pcm_bytes']}B, {info['elapsed_s']}s)"
                )

            try:
                if tts_backend == "google":
                    # Куски уходят в сокет сразу по готовности; следующий
                    # кусок качается, пока играется текущий.
                    async for chunk in stream_google_pcm(
                        speech, on_status=on_tts_status
                    ):
                        bytes_sent += len(chunk)
                        await send_bytes(chunk)
                else:
                    async for chunk in stream_pcm(
                        speech,
                        output_format=config.VOICE_TTS_FORMAT,
                        optimize_streaming_latency=config.VOICE_TTS_LATENCY,
                    ):
                        bytes_sent += len(chunk)
                        await send_bytes(chunk)
            except Exception as exc:
                logger.exception(
                    f"voice TTS streaming failed (backend={tts_backend})"
                )
                await send_json(
                    {"type": "error", "message": f"TTS error: {exc}"}
                )

            logger.info(
                f"voice {client_id}: answer {len(answer)} chars, "
                f"TTS backend={tts_backend}, "
                f"pieces={pieces_seen if tts_backend == 'google' else 1}, "
                f"{bytes_sent} bytes PCM"
            )
            await send_json(
                {"type": "done", "elapsed": time.time() - started}
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"voice turn failed for {client_id}")
            with suppress(Exception):
                await send_json(
                    {"type": "error", "message": "voice pipeline failed"}
                )
                await send_json(
                    {"type": "done", "elapsed": time.time() - started}
                )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue

            if "bytes" in message and message["bytes"] is not None:
                if not accept_audio:
                    continue
                for event in segmenter.feed(message["bytes"]):
                    if event["type"] == "speech_start":
                        await send_json(
                            {"type": "vad", "event": "speech_start"}
                        )
                    elif event["type"] == "speech_end":
                        accept_audio = False
                        await send_json(
                            {"type": "vad", "event": "speech_end"}
                        )
                        if turn_task is None or turn_task.done():
                            turn_task = asyncio.create_task(
                                run_turn(event["audio"]),
                                name=f"voice-turn-{client_id}",
                            )
                continue

            if "text" not in message or message["text"] is None:
                continue

            import json

            try:
                data = json.loads(message["text"])
            except Exception:
                continue
            action = data.get("action")

            # Браузерная сборка MAX WebApp может прислать initData. В MAX
            # браузерного клиента не пускаем, поэтому сообщение игнорируем.
            if data.get("type") == "auth":
                logger.debug(f"voice {client_id}: auth message ignored")
                continue

            if action == "listen_start":
                session_open = True
                if turn_task is not None and not turn_task.done():
                    turn_task.cancel()
                segmenter.reset()
                accept_audio = True
                logger.debug(f"voice {client_id}: listening")

            elif action == "stop":
                session_open = False
                accept_audio = False
                segmenter.reset()
                if turn_task is not None and not turn_task.done():
                    turn_task.cancel()
                logger.debug(f"voice {client_id}: stopped")

            elif action == "playback_done":
                if session_open and (turn_task is None or turn_task.done()):
                    segmenter.reset()
                    accept_audio = True
                    logger.debug(f"voice {client_id}: playback done, listening")

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception(f"voice socket failed for {client_id}")
    finally:
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
            with suppress(asyncio.CancelledError):
                await turn_task
        logger.info(f"voice client disconnected: {client_id!r} ({name})")


def _web_client_id(websocket: WebSocket) -> str:
    """Короткий идентификатор браузерного клиента: ?client=... или случайный."""
    requested = (websocket.query_params.get("client") or "").strip()
    safe = "".join(ch for ch in requested if ch.isalnum() or ch in "-_")[:32]
    return f"web:{safe or uuid.uuid4().hex[:8]}"


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    """Desktop-клиенты: обязательный HELLO с CLIENT_TOKEN."""
    await websocket.accept()

    # Страница /inysha/voice/ (сборка Vite) ходит именно сюда с ?dev=1 и не
    # умеет HELLO, поэтому для неё включаем браузерный режим без токена.
    # ?dev=1 — режим сравнения на ElevenLabs; без него — Google TTS.
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
    """Браузерный вариант: без HELLO и CLIENT_TOKEN (в MAX не пускаем)."""
    await websocket.accept()
    await _serve_voice(
        websocket, _web_client_id(websocket), "web", tts_backend="google"
    )

"""iNysha desktop client entry point.

Run on the machine with the screen, microphone and speakers:

    python -m client.main

The client keeps a control WebSocket (``/ws/client``) and a voice WebSocket
(``/ws/voice``) open.  Press the global hotkey (default F8) or type ``v`` in
the console to start/stop a voice session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import platform
import shutil
import subprocess
import sys
import uuid
from contextlib import suppress
from typing import Any

from loguru import logger
from rich.console import Console
from rich.prompt import Prompt

# Rich's prompt uses the builtin input(), which only supports cursor movement,
# Home/End and history when GNU readline is loaded.  On Windows the console
# already edits input natively, so only load it where it exists.
if sys.platform != "win32":
    try:
        import readline  # noqa: F401  # enables line editing for Prompt.ask
    except ImportError:
        pass

from client.audio import Microphone, PcmPlayer, list_devices, record_wav
from client.config import ClientConfig
from client.hotkey import start_hotkey, stop_hotkey
from client.screen import ScreenCapture

try:
    from websockets.asyncio.client import connect
except ImportError:  # websockets < 14
    from websockets import connect  # type: ignore


class DesktopClient:
    def __init__(self, cfg: ClientConfig) -> None:
        self.cfg = cfg
        self.console = Console()
        self.screen = ScreenCapture()
        self.mic: Microphone | None = None
        self.player: PcmPlayer | None = None
        self.control_ws = None
        self.voice_ws = None
        self.session_active = False
        self.mic_muted = True
        self.turn_had_audio = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping: asyncio.Event | None = None
        self._audio_queue: asyncio.Queue[bytes] | None = None
        self._control_send_lock = asyncio.Lock()
        self._rpc_tasks: set[asyncio.Task] = set()
        self._prompt_futures: dict[str, asyncio.Future[str]] = {}
        self._hotkey_listener = None
        self._server_client_id: str | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopping = asyncio.Event()
        self._audio_queue = asyncio.Queue(maxsize=100)

        logger.info(
            f"iNysha client '{self.cfg.client_id}' -> {self.cfg.server_url}"
        )
        self._hotkey_listener = start_hotkey(
            self._loop, self.toggle_session, self.cfg.hotkey
        )

        tasks = [
            asyncio.create_task(self._control_loop(), name="control"),
            asyncio.create_task(self._voice_loop(), name="voice"),
            asyncio.create_task(self._console_loop(), name="console"),
        ]
        if self.cfg.auto_start:
            await asyncio.sleep(1.0)
            await self.start_session()

        try:
            # The console task returns on 'q'; the socket tasks keep running
            # until shutdown, so wait for the first one to finish.
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self._stopping.set()
            stop_hotkey(self._hotkey_listener)
            await self.stop_session()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in list(self._rpc_tasks):
                task.cancel()

    # ------------------------------------------------------------------
    # control channel
    # ------------------------------------------------------------------
    async def _send_control(self, payload: dict[str, Any]) -> None:
        ws = self.control_ws
        if ws is None:
            raise RuntimeError("control socket is not connected")
        async with self._control_send_lock:
            await ws.send(json.dumps(payload, ensure_ascii=False))

    async def _hello(self) -> dict[str, Any]:
        return {
            "type": "hello",
            "token": self.cfg.token,
            "client_id": self.cfg.client_id,
            "name": self.cfg.name,
            "capabilities": {
                "screen": True,
                "mic": True,
                "speaker": True,
                "notify": True,
                **self.cfg.info(),
            },
        }

    async def _control_loop(self) -> None:
        url = f"{self.cfg.server_url}/ws/client"
        backoff = 1.0
        while not self._stopping.is_set():
            try:
                async with connect(
                    url,
                    max_size=64 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self.control_ws = ws
                    backoff = 1.0
                    await ws.send(json.dumps(await self._hello()))
                    logger.info("control channel connected")
                    async for raw in ws:
                        try:
                            message = json.loads(raw)
                        except Exception:
                            continue
                        if message.get("type") == "welcome":
                            self._server_client_id = message.get("client_id")
                            logger.info(
                                f"registered as '{self._server_client_id}' "
                                f"({message.get('name')})"
                            )
                        elif message.get("type") == "rpc":
                            task = asyncio.create_task(
                                self._handle_rpc(message),
                                name=f"rpc-{message.get('method')}",
                            )
                            self._rpc_tasks.add(task)
                            task.add_done_callback(self._rpc_tasks.discard)
                        elif message.get("type") == "prompt_result":
                            self._resolve_prompt(message)
                        elif message.get("type") == "pong":
                            pass
                        else:
                            logger.debug(f"control message: {message}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"control channel disconnected: {exc}")
            finally:
                self.control_ws = None
                self._fail_prompt_futures("control channel disconnected")
            if self._stopping.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    def _resolve_prompt(self, message: dict[str, Any]) -> None:
        prompt_id = str(message.get("id") or "")
        future = self._prompt_futures.get(prompt_id)
        if future is None or future.done():
            logger.warning(f"late/unknown prompt result: {prompt_id}")
            return
        if message.get("ok"):
            future.set_result(str(message.get("text") or ""))
        else:
            future.set_exception(
                RuntimeError(str(message.get("error") or "prompt failed"))
            )

    def _fail_prompt_futures(self, reason: str) -> None:
        for future in list(self._prompt_futures.values()):
            if not future.done():
                future.set_exception(ConnectionError(reason))

    async def _handle_rpc(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        try:
            result = await self._dispatch_rpc(str(method), params)
            reply = {
                "type": "rpc_result",
                "id": request_id,
                "ok": True,
                "result": result,
            }
        except Exception as exc:
            logger.exception(f"RPC {method} failed")
            reply = {
                "type": "rpc_result",
                "id": request_id,
                "ok": False,
                "error": str(exc),
            }
        try:
            await self._send_control(reply)
        except Exception as exc:
            logger.warning(f"could not send RPC result: {exc}")

    async def _dispatch_rpc(self, method: str, params: dict[str, Any]) -> Any:
        if method == "ping":
            return {"pong": True, "client_id": self.cfg.client_id}
        if method == "screenshot":
            return await asyncio.to_thread(
                self.screen.capture,
                int(params.get("monitor", 0)),
                float(params.get("resize_factor", 0.5)),
                str(params.get("format", "png")),
            )
        if method == "capture_sequence":
            return await asyncio.to_thread(
                self.screen.capture_sequence,
                int(params.get("monitor", 0)),
                float(params.get("interval_seconds", 0.5)),
                int(params.get("num_frames", 5)),
                float(params.get("resize_factor", 0.5)),
                str(params.get("format", "png")),
            )
        if method == "record_audio":
            if self.session_active:
                raise RuntimeError("mic is busy with an active voice session")
            duration = float(params.get("duration", 5.0))
            wav = await asyncio.to_thread(
                record_wav,
                duration,
                self.cfg.sample_rate,
                self.cfg.input_device,
            )
            return {
                "data": base64.b64encode(wav).decode("ascii"),
                "format": "wav",
                "duration": duration,
            }
        if method == "play_pcm":
            data = base64.b64decode(params["data"])
            sample_rate = int(params.get("sample_rate", self.cfg.sample_rate))
            if sample_rate != self.cfg.sample_rate:
                raise RuntimeError(
                    f"unsupported PCM rate {sample_rate}; client expects "
                    f"{self.cfg.sample_rate}"
                )
            await self._play_pcm(data)
            return {"ok": True, "bytes": len(data)}
        if method == "notify":
            await asyncio.to_thread(
                self._notify,
                str(params.get("title") or "iNysha"),
                str(params.get("message") or ""),
            )
            return {"ok": True}
        raise RuntimeError(f"unknown client method: {method}")

    async def _play_pcm(self, data: bytes) -> None:
        if self.player is None:
            self.player = PcmPlayer(
                sample_rate=self.cfg.sample_rate,
                device=self.cfg.output_device,
            )
            self.player.start()
        self.player.clear()
        self.player.enqueue(data)
        if not await self.player.wait_drained(timeout=180.0):
            raise RuntimeError("speaker playback timed out")

    @staticmethod
    def _notify(title: str, message: str) -> None:
        print(f"\n[{title}] {message}\n", flush=True)
        try:
            if sys.platform.startswith("linux") and shutil.which("notify-send"):
                subprocess.run(
                    ["notify-send", title, message],
                    timeout=3,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif sys.platform == "darwin" and shutil.which("osascript"):
                script = (
                    f'display notification {json.dumps(message)} '
                    f'with title {json.dumps(title)}'
                )
                subprocess.run(
                    ["osascript", "-e", script],
                    timeout=3,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # voice channel
    # ------------------------------------------------------------------
    async def _voice_loop(self) -> None:
        url = f"{self.cfg.server_url}/ws/voice"
        backoff = 1.0
        while not self._stopping.is_set():
            sender: asyncio.Task | None = None
            try:
                async with connect(
                    url,
                    max_size=16 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self.voice_ws = ws
                    backoff = 1.0
                    await ws.send(json.dumps(await self._hello()))
                    sender = asyncio.create_task(
                        self._voice_sender(ws), name="voice-sender"
                    )
                    logger.info("voice channel connected")
                    if self.session_active:
                        await ws.send(
                            json.dumps({"action": "listen_start"})
                        )
                        self.mic_muted = False
                    async for raw in ws:
                        if isinstance(raw, (bytes, bytearray)):
                            self._on_voice_bytes(bytes(raw))
                        else:
                            await self._on_voice_message(raw, ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"voice channel disconnected: {exc}")
            finally:
                self.voice_ws = None
                self.mic_muted = True
                if sender is not None:
                    sender.cancel()
                    with suppress(asyncio.CancelledError):
                        await sender
            if self._stopping.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _voice_sender(self, ws) -> None:
        assert self._audio_queue is not None
        while True:
            data = await self._audio_queue.get()
            try:
                await ws.send(data)
            except Exception:
                return

    def _on_voice_bytes(self, data: bytes) -> None:
        self.turn_had_audio = True
        if self.player is None:
            self.player = PcmPlayer(
                sample_rate=self.cfg.sample_rate,
                device=self.cfg.output_device,
            )
            self.player.start()
        self.player.enqueue(data)

    async def _on_voice_message(self, raw: str, ws) -> None:
        try:
            message = json.loads(raw)
        except Exception:
            return
        mtype = message.get("type")

        if mtype == "ready":
            logger.info(f"voice ready for '{message.get('name')}'")
            if self.session_active:
                await ws.send(json.dumps({"action": "listen_start"}))
                self.mic_muted = False
        elif mtype == "vad":
            if message.get("event") == "speech_start":
                print("🎤 ...", end="\r", flush=True)
            elif message.get("event") == "speech_end":
                self.mic_muted = True
                self.turn_had_audio = False
                print("… обрабатываю" + " " * 20)
        elif mtype == "status":
            status = message.get("status")
            if status == "stt":
                print("… распознаю" + " " * 20)
            elif status == "llm":
                print("… думаю" + " " * 20)
            elif status == "tts":
                print("… говорю" + " " * 20)
            elif status == "no_speech":
                print("Не расслышал." + " " * 20)
        elif mtype == "stt":
            print(f"\nТы: {message.get('text', '')}")
        elif mtype == "final":
            print(f"iNysha: {message.get('text', '')}")
        elif mtype == "error":
            print(f"\nОшибка: {message.get('message')}")
        elif mtype == "done":
            await self._finish_playback(ws)

    async def _finish_playback(self, ws) -> None:
        if self.turn_had_audio and self.player is not None:
            await self.player.wait_drained(timeout=180.0)
        self.turn_had_audio = False
        if self.session_active:
            self.mic_muted = False
            try:
                await ws.send(json.dumps({"action": "playback_done"}))
            except Exception:
                pass

    def _mic_callback(self, data: bytes) -> None:
        if (
            not self.session_active
            or self.mic_muted
            or self.voice_ws is None
            or self._loop is None
            or self._audio_queue is None
        ):
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue_mic, data)
        except RuntimeError:
            pass

    def _enqueue_mic(self, data: bytes) -> None:
        if self._audio_queue is None or self._audio_queue.full():
            return
        self._audio_queue.put_nowait(data)

    async def start_session(self) -> None:
        if self.session_active:
            return
        if self.voice_ws is None:
            print("Голосовой канал ещё не подключён.")
            return
        try:
            if self.player is None:
                self.player = PcmPlayer(
                    sample_rate=self.cfg.sample_rate,
                    device=self.cfg.output_device,
                )
                self.player.start()
            if self.mic is None:
                self.mic = Microphone(
                    sample_rate=self.cfg.sample_rate,
                    frame_ms=self.cfg.frame_ms,
                    device=self.cfg.input_device,
                )
            self.mic.start(self._mic_callback)
            self.session_active = True
            self.mic_muted = False
            if self._audio_queue is not None:
                while not self._audio_queue.empty():
                    with suppress(asyncio.QueueEmpty):
                        self._audio_queue.get_nowait()
            await self.voice_ws.send(json.dumps({"action": "listen_start"}))
            print(f"\n🎙  Слушаю... (хоткей {self.cfg.hotkey} или 'v' — стоп)")
        except Exception as exc:
            self.session_active = False
            self.mic_muted = True
            if self.mic is not None:
                self.mic.stop()
            print(f"Не удалось начать голосовую сессию: {exc}")

    async def stop_session(self) -> None:
        if not self.session_active and self.mic is None and self.player is None:
            return
        self.session_active = False
        self.mic_muted = True
        if self.voice_ws is not None:
            with suppress(Exception):
                await self.voice_ws.send(json.dumps({"action": "stop"}))
        if self.mic is not None:
            self.mic.stop()
            self.mic = None
        if self.player is not None:
            self.player.stop()
            self.player = None
        print("Голосовая сессия остановлена.")

    def toggle_session(self) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.stop_session() if self.session_active else self.start_session(),
            self._loop,
        )

    # ------------------------------------------------------------------
    # text prompts
    # ------------------------------------------------------------------
    async def send_prompt(self, text: str) -> None:
        """Send typed text to the shared agent and print its answer."""
        text = text.strip()
        if len(text) <= 1:
            return
        if self.control_ws is None:
            print("Сервер ещё не подключён — текст не отправлен.")
            return

        prompt_id = uuid.uuid4().hex
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._prompt_futures[prompt_id] = future
        try:
            await self._send_control(
                {"type": "prompt", "id": prompt_id, "text": text}
            )
            print("… отправлено агенту", flush=True)
            answer = await asyncio.wait_for(future, timeout=300.0)
            print(f"\niNysha: {answer}\n", flush=True)
        except asyncio.TimeoutError:
            print("\nАгент не ответил за 5 минут.")
        except Exception as exc:
            print(f"\nОшибка запроса: {exc}")
        finally:
            self._prompt_futures.pop(prompt_id, None)

    # ------------------------------------------------------------------
    # console
    # ------------------------------------------------------------------
    async def _console_loop(self) -> None:
        await asyncio.sleep(0.5)
        print(
            "\nКоманды: v — голос, s — скриншот, d — аудиоустройства, "
            "q — выход.\nЛюбой другой текст отправляется агенту.\n"
        )
        while not self._stopping.is_set():
            try:
                line = await asyncio.to_thread(
                    Prompt.ask, "iNysha", console=self.console
                )
            except (EOFError, OSError):
                await asyncio.sleep(1.0)
                continue
            raw = (line or "").strip()
            if not raw:
                continue
            command = raw.lower()
            if command in {"q", "quit", "exit"}:
                self._stopping.set()
                return
            if command in {"v", "voice"}:
                if self.session_active:
                    await self.stop_session()
                else:
                    await self.start_session()
            elif command in {"s", "screen", "screenshot"}:
                try:
                    shot = await asyncio.to_thread(self.screen.capture, 0, 0.5, "png")
                    print(
                        f"Скриншот: {shot['width']}x{shot['height']}, "
                        f"{shot['bytes']} bytes (base64)"
                    )
                except Exception as exc:
                    print(f"Ошибка скриншота: {exc}")
            elif command in {"d", "devices"}:
                try:
                    print(list_devices())
                except Exception as exc:
                    print(f"Ошибка аудио: {exc}")
            elif len(raw) > 1:
                await self.send_prompt(raw)
            else:
                print("Неизвестная команда. v / s / d / q")


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def main() -> None:
    configure_logging()
    cfg = ClientConfig.from_env()
    if not cfg.token:
        print(
            "CLIENT_TOKEN is not set. Put the same CLIENT_TOKEN as on the "
            "server into the client .env (or environment)."
        )
        raise SystemExit(2)
    print(f"Платформа: {platform.system()} {platform.release()}")
    client = DesktopClient(cfg)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

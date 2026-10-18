"""Cross-platform microphone capture and PCM playback via PortAudio.

``sounddevice`` wraps PortAudio on Linux and Windows, so the same code works
on both.  Audio is always 16 kHz / 16-bit / mono — the format the server's
WebRTC VAD and ElevenLabs ``pcm_16000`` stream expect.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Callable

from loguru import logger

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - depends on the desktop machine
    sd = None
    _AUDIO_IMPORT_ERROR = exc
else:
    _AUDIO_IMPORT_ERROR = None


def require_audio() -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice/PortAudio is not available. On Ubuntu install "
            "'libportaudio2'; on Windows install requirements-client.txt. "
            f"Import error: {_AUDIO_IMPORT_ERROR}"
        )


def list_devices() -> str:
    require_audio()
    return str(sd.query_devices())


def resample_pcm(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear resampling for devices that reject 16 kHz natively.

    Some Windows host APIs only expose 44.1/48 kHz.  The server still wants
    16 kHz, so the client converts at the edge.
    """
    if not data or src_rate == dst_rate:
        return data
    import numpy as np

    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return data
    target = max(1, int(round(samples.size * dst_rate / src_rate)))
    x_old = np.arange(samples.size, dtype=np.float64)
    x_new = np.linspace(0, samples.size - 1, target)
    converted = np.interp(x_new, x_old, samples).astype(np.int16)
    return converted.tobytes()


class Microphone:
    """Raw 16-bit mono microphone stream with a callback per block."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        device=None,
    ) -> None:
        require_audio()
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.blocksize = max(1, sample_rate * frame_ms // 1000)
        self.capture_rate = sample_rate
        self.device = device
        self._stream = None
        self._on_chunk: Callable[[bytes], None] | None = None

    def start(self, on_chunk: Callable[[bytes], None]) -> None:
        if self._stream is not None:
            return
        self._on_chunk = on_chunk
        last_error = None
        # Try the native 16 kHz first, then common Windows rates.
        for rate in (self.sample_rate, 48000, 44100):
            try:
                self.capture_rate = rate
                self._stream = sd.RawInputStream(
                    samplerate=rate,
                    blocksize=max(1, rate * self.frame_ms // 1000),
                    device=self.device,
                    channels=1,
                    dtype="int16",
                    latency="low",
                    callback=self._callback,
                )
                self._stream.start()
                if rate != self.sample_rate:
                    logger.info(
                        f"microphone opened at {rate} Hz; resampling to "
                        f"{self.sample_rate} Hz"
                    )
                return
            except Exception as exc:
                last_error = exc
                self._stream = None
        raise RuntimeError(
            f"Could not open microphone (device={self.device!r}): {last_error}"
        ) from last_error

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug(f"microphone status: {status}")
        if self._on_chunk is not None:
            self._on_chunk(
                resample_pcm(bytes(indata), self.capture_rate, self.sample_rate)
            )

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        self._on_chunk = None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()


class PcmPlayer:
    """Queue raw PCM chunks into one low-latency output stream.

    ``queue.join()`` semantics are used so the voice session only reports
    ``playback_done`` after the speaker has actually consumed the last chunk,
    not merely when the last network packet arrived.
    """

    def __init__(self, sample_rate: int = 16000, device=None) -> None:
        require_audio()
        self.sample_rate = sample_rate
        self.play_rate = sample_rate
        self.device = device
        self._stream = None
        self._thread = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._closed = False

    def start(self) -> None:
        if self._stream is not None:
            return
        last_error = None
        for rate in (self.sample_rate, 48000, 44100):
            try:
                self.play_rate = rate
                self._stream = sd.RawOutputStream(
                    samplerate=rate,
                    device=self.device,
                    channels=1,
                    dtype="int16",
                    latency="low",
                )
                self._stream.start()
                if rate != self.sample_rate:
                    logger.info(
                        f"speakers opened at {rate} Hz; resampling from "
                        f"{self.sample_rate} Hz"
                    )
                break
            except Exception as exc:
                last_error = exc
                self._stream = None
        if self._stream is None:
            raise RuntimeError(
                f"Could not open speakers (device={self.device!r}): {last_error}"
            ) from last_error
        self._closed = False
        self._thread = threading.Thread(
            target=self._writer, name="inysha-speaker", daemon=True
        )
        self._thread.start()

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._stream is not None:
                    self._stream.write(item)
            except Exception as exc:
                logger.error(f"speaker write failed: {exc}")
            finally:
                self._queue.task_done()

    def enqueue(self, data: bytes) -> None:
        if not data or self._stream is None or self._closed:
            return
        self._queue.put(
            resample_pcm(data, self.sample_rate, self.play_rate)
        )

    async def wait_drained(self, timeout: float | None = None) -> bool:
        """Wait until all queued audio has been written to the device."""
        if self._stream is None:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.02)
        return True

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._closed = True
        self.clear()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._stream.stop()
        finally:
            self._stream.close()
            self._stream = None


def record_wav(duration: float, sample_rate: int = 16000, device=None) -> bytes:
    """Record a fixed-duration WAV using PortAudio (blocking)."""
    require_audio()
    import io
    import wave

    last_error = None
    recording = None
    capture_rate = sample_rate
    for rate in (sample_rate, 48000, 44100):
        try:
            frames = int(duration * rate)
            recording = sd.rec(
                frames,
                samplerate=rate,
                channels=1,
                dtype="int16",
                device=device,
            )
            sd.wait()
            capture_rate = rate
            break
        except Exception as exc:
            last_error = exc
            recording = None
    if recording is None:
        raise RuntimeError(f"Could not record microphone: {last_error}") from last_error

    raw = recording.tobytes()
    if capture_rate != sample_rate:
        raw = resample_pcm(raw, capture_rate, sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return buf.getvalue()

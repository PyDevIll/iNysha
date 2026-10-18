"""WebRTC-based voice activity detection and utterance segmentation.

The desktop client streams raw 16-bit mono PCM continuously. This module chops it
into 30 ms frames, classifies each frame as speech/non-speech with
``webrtcvad``, and emits segment events:

    {"type": "speech_start"}
    {"type": "speech_end", "audio": <bytes>}

Segmentation rules:
  * A short *pre-roll* buffer is kept so the first (quiet) consonant of an
    utterance is not clipped.
  * Speech must last ``min_speech_ms`` before a segment starts (ignores clicks).
  * A segment ends after ``hangover_ms`` of consecutive non-speech frames
    (so a natural pause mid-sentence does not truncate the user).
  * Trailing silence is trimmed before the audio is handed to the pipeline.
  * ``max_speech_ms`` is a hard cap so a noisy mic can never grow unbounded.
"""

from __future__ import annotations

from collections import deque

import webrtcvad

SAMPLE_RATE = 16000


class VadSegmenter:
    def __init__(
        self,
        aggressiveness: int = 2,
        frame_ms: int = 30,
        preroll_ms: int = 400,
        min_speech_ms: int = 250,
        hangover_ms: int = 700,
        max_speech_ms: int = 20000,
    ) -> None:
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        # 16-bit mono samples.
        self.frame_bytes = SAMPLE_RATE * frame_ms // 1000 * 2

        self._preroll: deque[bytes] = deque(maxlen=max(1, preroll_ms // frame_ms))
        self._min_speech_frames = max(1, min_speech_ms // frame_ms)
        self._hangover_frames = max(1, hangover_ms // frame_ms)
        self._max_speech_frames = max(1, max_speech_ms // frame_ms)

        self._buffer = bytearray()
        self._voiced: list[bytes] = []
        self._in_speech = False
        self._voiced_run = 0
        self._silent_run = 0

    def reset(self) -> None:
        """Drop all buffered state (e.g. between turns or on disconnect)."""
        self._buffer.clear()
        self._preroll.clear()
        self._voiced = []
        self._in_speech = False
        self._voiced_run = 0
        self._silent_run = 0

    def has_pending_speech(self) -> bool:
        return self._in_speech

    def feed(self, pcm: bytes) -> list[dict]:
        """Feed arbitrary-length PCM; returns any events produced."""
        events: list[dict] = []
        self._buffer.extend(pcm)
        frame_bytes = self.frame_bytes
        while len(self._buffer) >= frame_bytes:
            frame = bytes(self._buffer[:frame_bytes])
            del self._buffer[:frame_bytes]
            events.extend(self._process_frame(frame))
        return events

    def flush(self) -> bytes | None:
        """Force-end an in-progress utterance (used for a manual stop)."""
        if not self._in_speech:
            return None
        return self._finish()

    def _process_frame(self, frame: bytes) -> list[dict]:
        is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

        if not self._in_speech:
            self._preroll.append(frame)
            if is_speech:
                self._voiced_run += 1
                if self._voiced_run >= self._min_speech_frames:
                    self._in_speech = True
                    self._silent_run = 0
                    # Include the pre-roll (and the speech onset it holds).
                    self._voiced = list(self._preroll)
                    return [{"type": "speech_start"}]
            else:
                self._voiced_run = 0
            return []

        self._voiced.append(frame)
        if is_speech:
            self._silent_run = 0
        else:
            self._silent_run += 1

        if (
            self._silent_run >= self._hangover_frames
            or len(self._voiced) >= self._max_speech_frames
        ):
            return [{"type": "speech_end", "audio": self._finish()}]
        return []

    def _finish(self) -> bytes:
        frames = self._voiced
        # Trim the hangover silence, but never below a minimal utterance.
        removable = max(0, len(frames) - self._min_speech_frames)
        trim = min(self._silent_run, removable)
        if trim:
            frames = frames[: len(frames) - trim]
        audio = b"".join(frames)
        self.reset()
        return audio

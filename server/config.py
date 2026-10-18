"""Central configuration for the iNysha cloud server.

All values come from the project ``.env`` so the legacy local scripts and the
cloud server agree on keys. Paths are absolute and created lazily.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
LOG_FILE = DATA_DIR / "agent.log"

# ── Server ─────────────────────────────────────────────────────────────
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8010"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── MAX ────────────────────────────────────────────────────────────────
MAX_BOT_TOKEN = os.getenv("MAX_BOT_TOKEN", "")
MAX_WEBHOOK_SECRET = os.getenv("MAX_WEBHOOK_SECRET", "")
MAX_REASONING_CHAT_ID = int(os.getenv("MAX_REASONING_CHAT_ID", "0") or "0")
MAX_SEND_REASONING = os.getenv("MAX_SEND_REASONING", "0").lower() in {"1", "true", "yes"}
# "poll" (default) or "webhook". Polling needs no public TLS certificate.
MAX_TRANSPORT = os.getenv("MAX_TRANSPORT", "poll").strip().lower()


def _id_set(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name, default)
    return {part.strip() for part in raw.split(",") if part.strip()}


# Mirrors the trusted sources listed in system_prompts/core.md.
TRUSTED_MAX_CHAT_IDS = _id_set("TRUSTED_MAX_CHAT_IDS", "326963375,367837204")
TRUSTED_MAX_USER_IDS = _id_set("TRUSTED_MAX_USER_IDS", "115302544,5156907")

# ── Desktop clients ────────────────────────────────────────────────────
# Shared secret for /ws/client and /ws/voice.  This is deliberately separate
# from MAX_WEBHOOK_SECRET: desktop clients must never be able to forge MAX
# webhook traffic.
CLIENT_TOKEN = os.getenv("CLIENT_TOKEN", "")
CLIENT_TRUSTED = os.getenv("CLIENT_TRUSTED", "1").lower() in {"1", "true", "yes"}

# ── Voice (used from Phase 3 onwards) ──────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ru")
VOICE_SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
VOICE_TURN_TIMEOUT = float(os.getenv("VOICE_TURN_TIMEOUT", "180"))
VOICE_MAX_SECONDS = int(os.getenv("VOICE_MAX_SECONDS", "20"))
# ElevenLabs PCM streaming.  `pcm_16000` is what the desktop client plays.
VOICE_TTS_FORMAT = os.getenv("VOICE_TTS_FORMAT", "pcm_16000")
VOICE_TTS_LATENCY = int(os.getenv("VOICE_TTS_LATENCY", "3"))

ELEVENLABS_API_KEY = (
    os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_TTS_API_KEY", "")
)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# ── Attachment safety limits ───────────────────────────────────────────
# Groq Whisper accepts up to 25 MB per request; leave a small margin.
MAX_AUDIO_MAX_BYTES = int(
    os.getenv("MAX_AUDIO_MAX_BYTES", str(24 * 1024 * 1024))
)
# General cap for non-audio attachments (images/video/files).
MAX_ATTACHMENT_MAX_BYTES = int(
    os.getenv("MAX_ATTACHMENT_MAX_BYTES", str(64 * 1024 * 1024))
)


def is_trusted(chat_id: str | None, user_id: str | None) -> bool:
    """Whether a MAX source is allowed to trigger restricted behaviour."""
    if chat_id and chat_id in TRUSTED_MAX_CHAT_IDS:
        return True
    if user_id and user_id in TRUSTED_MAX_USER_IDS:
        return True
    return False


def ensure_dirs() -> None:
    for path in (DATA_DIR, DOWNLOADS_DIR, SCREENSHOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)

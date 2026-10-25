"""Central configuration for one iNysha-compatible instance.

Everything instance-specific is read here from the checkout's ``.env``:
identity, public webhook route, secret header, data/static/prompt paths and
server settings.  Modules must import these values instead of hardcoding
"iНюша", "inysha", ``data/`` or a particular webhook URL.

To run a second instance from another directory, copy the checkout, copy
``.env.example`` to ``.env``, and change at least ``INSTANCE_SLUG`` (which
derives the MAX webhook path by default), the tokens, the port and — if the
new checkout should not share history — ``DATA_DIR``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _route_path(name: str, default: str) -> str:
    raw = os.getenv(name)
    value = default if raw is None else raw.strip()
    if not value:
        return ""
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/") or "/"


def _slug(value: str, default: str = "inysha") -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or default


# ── Instance identity ──────────────────────────────────────────────────
# INSTANCE_NAME is user-facing; INSTANCE_SLUG is the technical id and the
# default prefix for the MAX webhook path.
INSTANCE_NAME = _env_str("INSTANCE_NAME", "iНюша")
INSTANCE_SLUG = _slug(os.getenv("INSTANCE_SLUG", "inysha"))
AGENT_NAME = _env_str("AGENT_NAME", "DELIRIA")
AGENT_DISPLAY_NAME = _env_str("AGENT_DISPLAY_NAME", "Deliria")
AGENT_PERSONA_NAME = _env_str("AGENT_PERSONA_NAME", "Делирия Фора")
HELPER_NAME = _env_str("HELPER_NAME", "HELPER")
SERVER_TITLE = _env_str("SERVER_TITLE", "iNysha Cloud")

# ── Public routes and shared-secret header ─────────────────────────────
# The prefixed route is the canonical MAX webhook; the short alias keeps the
# old unprefixed endpoint working. Set MAX_WEBHOOK_ALIAS_PATH="" to disable it.
MAX_WEBHOOK_PATH = _route_path(
    "MAX_WEBHOOK_PATH", f"/{INSTANCE_SLUG}/max-webhook"
) or f"/{INSTANCE_SLUG}/max-webhook"
MAX_WEBHOOK_ALIAS_PATH = _route_path(
    "MAX_WEBHOOK_ALIAS_PATH", "/max-webhook"
)
SECRET_HEADER = _env_str("SECRET_HEADER", "X-Inysha-Secret")

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = _env_path("DATA_DIR", ROOT / "data")
DOWNLOADS_DIR = _env_path("DOWNLOADS_DIR", DATA_DIR / "downloads")
SCREENSHOTS_DIR = _env_path("SCREENSHOTS_DIR", DATA_DIR / "screenshots")
LOG_FILE = _env_path("LOG_FILE", DATA_DIR / "agent.log")
TTS_CACHE_DIR = _env_path("TTS_CACHE_DIR", DATA_DIR / "tts_cache")
SCHEDULER_TASKS_FILE = _env_path(
    "SCHEDULER_TASKS_FILE", DATA_DIR / "scheduler_tasks.json"
)
MAX_MARKER_FILE = _env_path("MAX_MARKER_FILE", DATA_DIR / "max_marker.json")

STATIC_DIR = _env_path("STATIC_DIR", ROOT / "static")
VOICE_STATIC_DIR = _env_path("VOICE_STATIC_DIR", STATIC_DIR / "voice")
PROMPTS_DIR = _env_path("PROMPTS_DIR", ROOT / "system_prompts")
CORE_PROMPT_PATH = PROMPTS_DIR / "core.md"
EXTENDED_PROMPT_PATH = PROMPTS_DIR / "extended.md"
TOOLS_GUIDELINES_PROMPT_PATH = PROMPTS_DIR / "tools_guidelines.md"

# ── Server ─────────────────────────────────────────────────────────────
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8010"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── LLM and search ─────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_KEY_HELPER = os.getenv("DEEPSEEK_API_KEY_HELPER", "")
DEEPSEEK_BASE_URL = (
    os.getenv("DEEPSEEK_BASE_URL")
    or os.getenv("DEEPSEEK_API_ENDPOINT")
    or "https://api.deepseek.com"
).rstrip("/")
DEEPSEEK_MODEL = (os.getenv("DEEPSEEK_MODEL") or "deepseek-flash").strip().replace(
    "_", "-"
) or "deepseek-flash"
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "30000"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── MAX ────────────────────────────────────────────────────────────────
MAX_BOT_TOKEN = os.getenv("MAX_BOT_TOKEN", "")
MAX_WEBHOOK_SECRET = os.getenv("MAX_WEBHOOK_SECRET", "")
MAX_REASONING_CHAT_ID = int(os.getenv("MAX_REASONING_CHAT_ID", "0") or "0")
MAX_SEND_REASONING = os.getenv("MAX_SEND_REASONING", "0").lower() in {"1", "true", "yes"}
# "poll" (default) or "webhook". Polling needs no public TLS certificate.
MAX_TRANSPORT = os.getenv("MAX_TRANSPORT", "poll").strip().lower()

# ── MAX batching ───────────────────────────────────────────────────────
# Сообщения одного чата копятся, пока не пройдёт MAX_DEFER_SECONDS тишины,
# и уходят в очередь одним запросом. Переполнение пачки сбрасывается сразу.
MAX_DEFER_SECONDS = float(os.getenv("MAX_DEFER_SECONDS", "6.0"))
MAX_BATCH_MESSAGES = int(os.getenv("MAX_BATCH_MESSAGES", "10"))
MAX_BATCH_CHARS = int(os.getenv("MAX_BATCH_CHARS", "4000"))

# Защита от лавины: сколько апдейтов разбираем одновременно (скачивание
# вложений и распознавание голоса идут до батчинга и стоят дорого).
MAX_CONCURRENT_UPDATES = int(os.getenv("MAX_CONCURRENT_UPDATES", "5"))
# Rate-limit на чат: не больше N сообщений за окно (секунды). 0 — выключить.
MAX_CHAT_RATE_LIMIT = int(os.getenv("MAX_CHAT_RATE_LIMIT", "10"))
MAX_CHAT_RATE_WINDOW = float(os.getenv("MAX_CHAT_RATE_WINDOW", "60"))


def _id_set(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name, default)
    return {part.strip() for part in raw.split(",") if part.strip()}


# Trusted MAX sources. The agent receives a computed ``trusted=true/false``
# flag in each message header; these values are not duplicated in prompts.
TRUSTED_MAX_CHAT_IDS = _id_set("TRUSTED_MAX_CHAT_IDS", "326963375,367837204")
TRUSTED_MAX_USER_IDS = _id_set("TRUSTED_MAX_USER_IDS", "115302544,5156907")

# ── Desktop clients ────────────────────────────────────────────────────
# Shared secret for /ws/client and /ws/voice.  This is deliberately separate
# from MAX_WEBHOOK_SECRET: desktop clients must never be able to forge MAX
# webhook traffic.
CLIENT_TOKEN = os.getenv("CLIENT_TOKEN", "")
CLIENT_TRUSTED = os.getenv("CLIENT_TRUSTED", "1").lower() in {"1", "true", "yes"}

# ── Voice ──────────────────────────────────────────────────────────────
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
ELEVENLABS_MAX_CHARS = int(os.getenv("ELEVENLABS_MAX_CHARS", "4000"))
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.2"))

# ── Attachment safety limits ───────────────────────────────────────────
# Groq Whisper accepts up to 25 MB per request; leave a small margin.
MAX_AUDIO_MAX_BYTES = int(
    os.getenv("MAX_AUDIO_MAX_BYTES", str(24 * 1024 * 1024))
)
# General cap for non-audio attachments (images/video/files).
MAX_ATTACHMENT_MAX_BYTES = int(
    os.getenv("MAX_ATTACHMENT_MAX_BYTES", str(64 * 1024 * 1024))
)


def prompt_substitutions() -> dict[str, str]:
    """Values available to prompt files as ``{{NAME}}`` placeholders.

    Prompt files stay readable, while instance-specific paths, names and
    trusted-id lists come from this central config.
    """
    return {
        "{{PROJECT_ROOT}}": str(ROOT),
        "{{DATA_DIR}}": str(DATA_DIR),
        "{{PROMPTS_DIR}}": str(PROMPTS_DIR),
        "{{INSTANCE_NAME}}": INSTANCE_NAME,
        "{{INSTANCE_SLUG}}": INSTANCE_SLUG,
        "{{AGENT_DISPLAY_NAME}}": AGENT_DISPLAY_NAME,
        "{{AGENT_PERSONA_NAME}}": AGENT_PERSONA_NAME,
        "{{MAX_WEBHOOK_PATH}}": MAX_WEBHOOK_PATH,
        "{{SERVER_PORT}}": str(SERVER_PORT),
        "{{TRUSTED_MAX_CHAT_IDS}}": ", ".join(sorted(TRUSTED_MAX_CHAT_IDS))
        or "(none)",
        "{{TRUSTED_MAX_USER_IDS}}": ", ".join(sorted(TRUSTED_MAX_USER_IDS))
        or "(none)",
    }


def is_trusted(chat_id: str | None, user_id: str | None) -> bool:
    """Whether a MAX source is allowed to trigger restricted behaviour."""
    if chat_id and chat_id in TRUSTED_MAX_CHAT_IDS:
        return True
    if user_id and user_id in TRUSTED_MAX_USER_IDS:
        return True
    return False


def ensure_dirs() -> None:
    for path in (
        DATA_DIR,
        DOWNLOADS_DIR,
        SCREENSHOTS_DIR,
        TTS_CACHE_DIR,
        LOG_FILE.parent,
        SCHEDULER_TASKS_FILE.parent,
        MAX_MARKER_FILE.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

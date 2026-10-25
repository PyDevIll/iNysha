"""Client configuration from environment / .env.

Only the client needs these values.  The server's ``CLIENT_TOKEN`` must match.
"""

from __future__ import annotations

import os
import platform
import re
import socket
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
# A client-local .env wins over the checkout's server .env, so a desktop copy
# can carry only SERVER_URL/CLIENT_TOKEN without the server secrets.
load_dotenv(PACKAGE_DIR / ".env")
load_dotenv(ROOT / ".env")
load_dotenv(Path.cwd() / ".env")


def _device(value: str | None):
    if not value:
        return None
    return int(value) if value.isdigit() else value


SERVER_PATH_PREFIX = os.getenv(
    "SERVER_PATH_PREFIX", os.getenv("INSTANCE_SLUG", "inysha")
).strip("/")
DEFAULT_SERVER_URL = f"ws://127.0.0.1:{os.getenv('SERVER_PORT', '8010')}"


def _flat_ws_url(raw: str) -> str:
    url = (raw or DEFAULT_SERVER_URL).strip().rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    # Be forgiving of the old nginx-style prefix from the first design.
    # Set SERVER_PATH_PREFIX (e.g. "delirya") for another instance path.
    if SERVER_PATH_PREFIX and url.endswith("/" + SERVER_PATH_PREFIX):
        url = url[: -len("/" + SERVER_PATH_PREFIX)]
    return url


@dataclass(frozen=True)
class ClientConfig:
    server_url: str
    token: str
    client_id: str
    name: str
    app_name: str
    hotkey: str
    sample_rate: int
    frame_ms: int
    input_device: object | None
    output_device: object | None
    auto_start: bool

    @classmethod
    def from_env(cls) -> "ClientConfig":
        host = socket.gethostname() or "desktop"
        raw_id = (
            os.getenv("CLIENT_ID")
            or host.lower().replace(" ", "-")
            or f"client-{os.getpid()}"
        )
        client_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_id).strip("-")[:64]
        return cls(
            server_url=_flat_ws_url(
                os.getenv("SERVER_URL", DEFAULT_SERVER_URL)
            ),
            token=os.getenv("CLIENT_TOKEN", ""),
            client_id=client_id or f"client-{os.getpid()}",
            name=os.getenv("CLIENT_NAME") or host,
            app_name=(
                os.getenv("CLIENT_APP_NAME")
                or os.getenv("INSTANCE_NAME")
                or "iNysha"
            ),
            hotkey=os.getenv("CLIENT_HOTKEY", "<f8>"),
            sample_rate=int(os.getenv("VOICE_SAMPLE_RATE", "16000")),
            frame_ms=int(os.getenv("CLIENT_FRAME_MS", "30")),
            input_device=_device(os.getenv("CLIENT_INPUT_DEVICE")),
            output_device=_device(os.getenv("CLIENT_OUTPUT_DEVICE")),
            auto_start=os.getenv("CLIENT_AUTO_START", "0").lower()
            in {"1", "true", "yes"},
        )

    def info(self) -> dict:
        return {
            "os": platform.system(),
            "release": platform.release(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
        }

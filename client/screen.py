"""Cross-platform screen capture with mss + Pillow."""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

try:
    import mss
except Exception as exc:  # pragma: no cover - depends on desktop
    mss = None
    _MSS_IMPORT_ERROR = exc
else:
    _MSS_IMPORT_ERROR = None


def require_screen() -> None:
    if mss is None:
        raise RuntimeError(f"mss is not available: {_MSS_IMPORT_ERROR}")


class ScreenCapture:
    """mss is not thread-safe; each worker thread gets its own instance."""

    def __init__(self) -> None:
        require_screen()
        self._local = threading.local()

    def _sct(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = mss.mss()
            self._local.sct = sct
        return sct

    def _capture_one(
        self,
        monitor: int,
        resize_factor: float,
        fmt: str,
    ) -> dict:
        try:
            sct = self._sct()
            index = int(monitor)
            if index < 0 or index >= len(sct.monitors):
                index = 1 if len(sct.monitors) > 1 else 0
            raw = sct.grab(sct.monitors[index])
            image = Image.frombytes("RGB", raw.size, raw.rgb)
        except Exception as exc:
            # mss needs X11.  On a Wayland session, fall back to the desktop
            # screenshot helper if one is installed.
            if not sys.platform.startswith("linux"):
                raise
            image = self._linux_fallback(exc)
        factor = max(0.1, min(float(resize_factor), 1.0))
        if factor != 1.0:
            image = image.resize(
                (max(1, int(image.width * factor)), max(1, int(image.height * factor))),
                Image.Resampling.LANCZOS,
            )
        out = io.BytesIO()
        if fmt.lower() in {"jpg", "jpeg"}:
            image.save(out, format="JPEG", quality=82, optimize=False)
            mime = "image/jpeg"
        else:
            image.save(out, format="PNG", compress_level=1)
            mime = "image/png"
        data = out.getvalue()
        return {
            "data": base64.b64encode(data).decode("ascii"),
            "mime": mime,
            "width": image.width,
            "height": image.height,
            "bytes": len(data),
        }

    def _linux_fallback(self, original: Exception) -> Image.Image:
        """Capture via grim/gnome-screenshot when mss cannot use X11."""
        commands: list[tuple[str, str]] = []
        if shutil.which("grim"):
            commands.append(("stdout", "grim"))
        if shutil.which("gnome-screenshot"):
            commands.append(("file", "gnome-screenshot"))
        if shutil.which("spectacle"):
            commands.append(("file", "spectacle"))

        for mode, program in commands:
            try:
                with tempfile.TemporaryDirectory(prefix="inysha-shot-") as tmp:
                    target = Path(tmp) / "shot.png"
                    if mode == "stdout":
                        command = [program, "-"]
                    elif program == "spectacle":
                        command = [program, "-b", "-n", "-o", str(target)]
                    else:
                        command = [program, "-f", str(target)]
                    run = subprocess.run(
                        command, capture_output=True, timeout=10
                    )
                    raw = run.stdout if mode == "stdout" else (
                        target.read_bytes() if target.exists() else b""
                    )
                    if run.returncode == 0 and raw:
                        return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
        raise RuntimeError(
            "screen capture failed: mss needs X11, and no Wayland helper "
            "(grim / gnome-screenshot / spectacle) worked. "
            f"Original error: {original}"
        )

    def capture(
        self,
        monitor: int = 0,
        resize_factor: float = 0.5,
        fmt: str = "png",
    ) -> dict:
        return self._capture_one(monitor, resize_factor, fmt)

    def capture_sequence(
        self,
        monitor: int = 0,
        interval_seconds: float = 0.5,
        num_frames: int = 5,
        resize_factor: float = 0.5,
        fmt: str = "png",
    ) -> dict:
        frames = []
        count = max(1, min(int(num_frames), 10))
        interval = max(0.0, float(interval_seconds))
        for i in range(count):
            frames.append(self._capture_one(monitor, resize_factor, fmt))
            if i < count - 1 and interval:
                time.sleep(interval)
        return {"frames": frames}

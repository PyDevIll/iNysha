"""Global hotkey via pynput (Windows and Linux/X11)."""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger


def start_hotkey(
    loop: asyncio.AbstractEventLoop,
    callback: Callable[[], None],
    hotkey: str = "<f8>",
):
    """Start a global hotkey listener; return it, or ``None`` if unavailable.

    On Linux, pynput needs an X11 session.  Wayland compositors usually block
    global key grabs; the console command ``v`` remains available there.
    """
    try:
        from pynput import keyboard
    except Exception as exc:
        logger.warning(
            f"global hotkey unavailable ({exc}); use the console command 'v'"
        )
        return None

    def fire() -> None:
        try:
            loop.call_soon_threadsafe(callback)
        except RuntimeError:
            # Loop already closed during shutdown.
            pass

    try:
        listener = keyboard.GlobalHotKeys({hotkey: fire})
        listener.start()
        logger.info(f"global hotkey active: {hotkey}")
        return listener
    except Exception as exc:
        logger.warning(
            f"could not register hotkey {hotkey!r}: {exc}; use console 'v'"
        )
        return None


def stop_hotkey(listener) -> None:
    if listener is None:
        return
    try:
        listener.stop()
    except Exception:
        pass

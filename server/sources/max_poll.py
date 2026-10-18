"""MAX long-polling transport (used when no webhook is available).

No public TLS certificate is required: the server calls ``GET /updates`` with
the bot token and feeds the results through the same dedup/processing path as
the webhook.
"""

from __future__ import annotations

import asyncio
import json

from loguru import logger

from server import config
from server.sources.max import handle_update

MARKER_FILE = config.DATA_DIR / "max_marker.json"


def _load_marker() -> int | None:
    try:
        if MARKER_FILE.exists():
            value = json.loads(MARKER_FILE.read_text(encoding="utf-8")).get("marker")
            return int(value) if value is not None else None
    except Exception:
        logger.warning("Could not read the MAX marker file", exc_info=True)
    return None


def _save_marker(marker: int | None) -> None:
    if marker is None:
        return
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        MARKER_FILE.write_text(
            json.dumps({"marker": marker}), encoding="utf-8"
        )
    except Exception:
        logger.warning("Could not save the MAX marker", exc_info=True)


async def poll_max_updates(rt) -> None:
    """Long-poll MAX and enqueue every new update."""
    if rt.bot is None:
        logger.error("MAX poller started without a bot")
        return

    marker = _load_marker()
    if marker is None:
        # First run: consume the backlog once and remember where we are, so a
        # restart does not replay old messages.
        try:
            res = await rt.bot._api._request(
                "GET", "/updates", query={"timeout": 0, "limit": 100}
            )
            marker = res.get("marker") if isinstance(res, dict) else None
            rt.bot._marker = marker
            _save_marker(marker)
            logger.info(f"MAX poller primed at marker {marker}")
        except Exception:
            logger.exception("MAX poller priming failed; starting from scratch")
    else:
        rt.bot._marker = marker
        logger.info(f"MAX poller resuming from marker {marker}")

    while True:
        try:
            result = await rt.bot._poll_updates(types="message_created")
            if result.get("ok"):
                updates = result.get("result") or []
                logger.debug(
                    f"MAX poll: {len(updates)} update(s), marker={rt.bot._marker}"
                )
                for update in updates:
                    try:
                        await handle_update(update)
                    except Exception:
                        logger.exception("MAX update handling failed")
                _save_marker(rt.bot._marker)
            else:
                logger.warning(f"MAX polling failed: {result.get('error')}")
                await asyncio.sleep(5)
                continue
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("MAX polling loop error")
            await asyncio.sleep(5)
            continue

        # The API requires at least 300 ms between long-poll calls.
        await asyncio.sleep(0.3)

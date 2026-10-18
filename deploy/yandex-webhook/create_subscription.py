"""Create (or replace) the MAX webhook subscription for the Yandex function.

Run from the project root after the new function version is deployed:

    cd ~/iNysha
    MAX_SUBSCRIPTION_SECRET='inysha_voice_2026' \
        .venv/bin/python deploy/yandex-webhook/create_subscription.py

The secret must match ``MAX_SUBSCRIPTION_SECRET`` in the function environment.
It must be 5-256 characters and contain only ``[A-Za-z0-9_-]``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from lib.max_bot import MAXBot  # noqa: E402

FUNCTION_URL = os.getenv(
    "MAX_FUNCTION_URL",
    "https://functions.yandexcloud.net/d4e8obdfp90a4tbbr6rp",
)
SUBSCRIPTION_SECRET = os.getenv("MAX_SUBSCRIPTION_SECRET", "")


async def main() -> None:
    if not SUBSCRIPTION_SECRET:
        raise SystemExit(
            "Set MAX_SUBSCRIPTION_SECRET to the same value used in the "
            "Yandex function environment."
        )

    bot = MAXBot(token=os.environ["MAX_BOT_TOKEN"])
    try:
        subs = await bot._api.get_subscriptions()
        for sub in subs.get("subscriptions") or []:
            print(f"deleting old subscription: {sub.get('url')}")
            print(await bot._api.delete_subscription(sub["url"]))

        print(f"creating subscription: {FUNCTION_URL}")
        result = await bot._api.create_subscription(
            FUNCTION_URL,
            update_types=["message_created"],
            secret=SUBSCRIPTION_SECRET,
        )
        print(result)
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())

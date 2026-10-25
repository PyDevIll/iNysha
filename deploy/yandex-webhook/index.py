"""Yandex Cloud Function: MAX webhook -> iNysha VDS over plain HTTP.

Deploy as a Python 3.11/3.12 function with entrypoint ``index.handler``.

Environment variables:
    FORWARD_URL           http://2.24.160.163:8010/inysha/max-webhook
    MAX_WEBHOOK_SECRET    same value as in the server's .env
    FORWARD_SECRET_HEADER forwarded secret header (default X-Inysha-Secret)
    DEBUG_PING_URL        optional: http://2.24.160.163:8010/api/function-ping

MAX sends its subscription secret in ``X-Max-Bot-Api-Secret``.  The function
validates it and forwards the same value in ``FORWARD_SECRET_HEADER``, which
must match ``SECRET_HEADER`` in the server's config.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

FORWARD_URL = os.environ.get("FORWARD_URL", "").strip()
SECRET = os.environ.get("MAX_WEBHOOK_SECRET", "")
FORWARD_SECRET_HEADER = (
    os.environ.get("FORWARD_SECRET_HEADER", "").strip()
    or "X-Inysha-Secret"
)
DEBUG_PING_URL = os.environ.get("DEBUG_PING_URL", "").strip()


def handler(event: dict, context: object) -> dict:
    if not FORWARD_URL:
        return {"statusCode": 500, "body": "FORWARD_URL is not set"}

    headers = {
        str(key).lower(): str(value)
        for key, value in (event.get("headers") or {}).items()
    }

    if SECRET:
        received = headers.get("x-max-bot-api-secret", "")
        if received != SECRET:
            return {"statusCode": 403, "body": "Invalid MAX secret"}

    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")

    try:
        update = json.loads(body)
    except Exception as exc:
        return {"statusCode": 400, "body": f"Invalid JSON: {exc}"}

    if DEBUG_PING_URL:
        message = update.get("message") or {}
        body_obj = message.get("body") or {}
        query = urllib.parse.urlencode(
            {
                "from": "yandex-function",
                "update_type": update.get("update_type", ""),
                "mid": body_obj.get("mid", ""),
                "body_len": len(body),
            }
        )
        ping_url = DEBUG_PING_URL + ("&" if "?" in DEBUG_PING_URL else "?") + query
        try:
            with urllib.request.urlopen(ping_url, timeout=3) as response:
                response.read()
        except Exception:
            pass

    request = urllib.request.Request(
        FORWARD_URL,
        data=body.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            FORWARD_SECRET_HEADER: SECRET,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        return {"statusCode": 200, "body": "OK"}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return {"statusCode": 502, "body": f"VDS HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return {"statusCode": 502, "body": f"Forward failed: {exc}"}

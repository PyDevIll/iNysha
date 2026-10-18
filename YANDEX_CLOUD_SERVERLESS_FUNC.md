## serverless function URL

```
https://functions.yandexcloud.net/d4e8obdfp90a4tbbr6rp
```

## Why this function exists

Native MAX voice messages are broken in **long polling**: MAX sends an empty
`message_created` update and the voice message is absent from
`GET /messages`. The same message is delivered correctly via **webhook**.

MAX requires a valid public TLS certificate for a webhook, but this VDS uses
HTTP on port 8010. The function is the public HTTPS endpoint; it forwards the
raw update to the VDS over plain HTTP.

## Function code

Runtime: **Python 3.11/3.12**. Entrypoint: **`index.handler`**.

```python
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

FORWARD_URL = os.environ.get("FORWARD_URL", "").strip()
SECRET = os.environ.get("MAX_WEBHOOK_SECRET", "")
DEBUG_PING_URL = os.environ.get("DEBUG_PING_URL", "").strip()


def handler(event, context):
    if not FORWARD_URL:
        return {"statusCode": 500, "body": "FORWARD_URL is not set"}

    headers = {
        str(key).lower(): str(value)
        for key, value in (event.get("headers") or {}).items()
    }

    # MAX sends this header because the subscription was created with a
    # secret. Reject everything else so the public function URL cannot be
    # used to inject fake updates.
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

    # Diagnostic ping. It proves the function was invoked even when the
    # forward to the VDS fails. Remove DEBUG_PING_URL after testing if you
    # do not want an extra request per update.
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
            "X-Inysha-Secret": SECRET,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        return {"statusCode": 200, "body": "OK"}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        # 502 makes MAX retry; the VDS deduplicates by mid.
        return {"statusCode": 502, "body": f"VDS HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return {"statusCode": 502, "body": f"Forward failed: {exc}"}
```

## Cloud function environment variables

| Variable | Value |
|---|---|
| `FORWARD_URL` | `http://2.24.160.163:8010/inysha/max-webhook` |
| `MAX_WEBHOOK_SECRET` | the exact value of `MAX_WEBHOOK_SECRET` from `~/iNysha/.env` |
| `DEBUG_PING_URL` | `http://2.24.160.163:8010/api/function-ping` (optional, for testing) |

`MAX_BOT_TOKEN` is **not needed** by the function.

The webhook subscription secret is the same `MAX_WEBHOOK_SECRET` value. MAX
sends it in `X-Max-Bot-Api-Secret`; the function checks it and forwards the
same value in `X-Inysha-Secret`, which the VDS accepts.

## VDS prerequisites

- `uvicorn` must listen on **`0.0.0.0:8010`** (plain HTTP, no nginx/TLS):

  ```bash
  cd ~/iNysha
  .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8010 --log-level debug
  ```

- The FastAPI app exposes both `/max-webhook` and `/inysha/max-webhook`.
- Port 8010 must be reachable from the internet (firewall/security group).
- `MAX_TRANSPORT=webhook` in `~/iNysha/.env`.
- Nginx/TLS is not used for this path.

## Verify the MAX subscription

The subscription is already set. Check it without printing the token:

```bash
cd ~/iNysha
TOKEN=$(grep -m1 '^MAX_BOT_TOKEN=' .env | cut -d= -f2-)
curl -s https://platform-api2.max.ru/subscriptions \
  -H "Authorization: $TOKEN"
```

Expected:

```json
{"subscriptions":[{"url":"https://functions.yandexcloud.net/d4e8obdfp90a4tbbr6rp","update_types":["message_created"]}]}
```

To recreate it (this replaces the existing subscription):

```bash
cd ~/iNysha
TOKEN=$(grep -m1 '^MAX_BOT_TOKEN=' .env | cut -d= -f2-)
SECRET=$(grep -m1 '^MAX_WEBHOOK_SECRET=' .env | cut -d= -f2-)
curl -s -X POST https://platform-api2.max.ru/subscriptions \
  -H "Authorization: $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"https://functions.yandexcloud.net/d4e8obdfp90a4tbbr6rp\",\"update_types\":[\"message_created\"],\"secret\":\"$SECRET\"}"
```

## Test 1 — function → VDS

Invoke the function directly with a synthetic MAX update:

```bash
SECRET=$(grep -m1 '^MAX_WEBHOOK_SECRET=' ~/iNysha/.env | cut -d= -f2-)
curl -s -X POST "https://functions.yandexcloud.net/d4e8obdfp90a4tbbr6rp" \
  -H "Content-Type: application/json" \
  -H "X-Max-Bot-Api-Secret: $SECRET" \
  -d '{
    "update_type": "message_created",
    "chat_id": 999000111,
    "message": {
      "body": {"mid": "function-test-1", "text": "function test"},
      "sender": {"user_id": 999000222, "name": "Function Test"},
      "recipient": {"chat_id": 999000111}
    }
  }'
```

Expected:

- Function response: `{"statusCode": 200, "body": "OK"}`
- VDS log: `Yandex function ping: ...`
- VDS log: `[<id>] queued MAX message from chat 999000111`
- The bot tries to reply to `999000111` and gets `chat.not.found` — that is
  expected for the fake chat.

If the function returns `502` or the ping does not appear, the function did
not reach the VDS. If the function returns `403`, the subscription secret and
`MAX_WEBHOOK_SECRET` do not match.

## Test 2 — MAX → function → VDS

1. Start the VDS server on `0.0.0.0:8010` with `MAX_TRANSPORT=webhook`.
2. Send a native voice message to the bot in MAX.
3. Expected VDS log:

```
Yandex function ping: {'from': 'yandex-function', 'update_type': 'message_created', 'mid': '...'}
MAX attachments: [{"type": "audio", ... "has_url": true, "has_token": true}]
MAX voice transcribed (...): '...'
[<id>] queued MAX message from chat ...
```

If the ping appears but no `MAX attachments` line follows, the function
received the update but the forwarded body was empty or forwarding failed.
If neither appears, MAX did not deliver to the function (subscription or
secret problem).

## Notes

- A native voice message is still a known MAX limitation in long polling;
  webhook is the workaround.
- Forwarded voice messages (`link.type = "forward"`) lose the audio
  attachment inside MAX itself; that is a separate limitation.
- Audio recorded and sent as a **file** (paperclip) arrives as
  `type: "file"` and works even via long polling.
- Do not run long polling and webhook at the same time.

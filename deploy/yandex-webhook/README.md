# MAX webhook via Yandex Cloud Function (HTTP only)

Native MAX voice messages are currently broken in **long polling**: MAX sends
an empty `message_created` update and the voice message is absent from
`GET /messages`. Webhook delivery works.

This VDS uses plain HTTP on port 8010. The Yandex function is the public
HTTPS endpoint required by MAX and forwards updates to
`http://2.24.160.163:8010/inysha/max-webhook`.

The authoritative copy of the function code and deployment steps is
[`../../YANDEX_CLOUD_SERVERLESS_FUNC.md`](../../YANDEX_CLOUD_SERVERLESS_FUNC.md).
The same code is in [`index.py`](index.py).

## Function environment

| Variable | Value |
|---|---|
| `FORWARD_URL` | `http://2.24.160.163:8010/inysha/max-webhook` |
| `MAX_WEBHOOK_SECRET` | same value as in `~/iNysha/.env` |
| `DEBUG_PING_URL` | `http://2.24.160.163:8010/api/function-ping` (optional) |

Entrypoint: `index.handler`. Runtime: Python 3.11/3.12.

## VDS

```bash
cd ~/iNysha
# .env: MAX_TRANSPORT=webhook
.venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8010 --log-level debug
```

No nginx or TLS is used for this path. Port 8010 must be reachable from the
internet.

## Subscription

The subscription URL is the function URL and the secret is
`MAX_WEBHOOK_SECRET`:

```bash
cd ~/iNysha
TOKEN=$(grep -m1 '^MAX_BOT_TOKEN=' .env | cut -d= -f2-)
SECRET=$(grep -m1 '^MAX_WEBHOOK_SECRET=' .env | cut -d= -f2-)
curl -s -X POST https://platform-api2.max.ru/subscriptions \
  -H "Authorization: $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"https://functions.yandexcloud.net/d4e8obdfp90a4tbbr6rp\",\"update_types\":[\"message_created\"],\"secret\":\"$SECRET\"}"
```

Do not run long polling and webhook at the same time.

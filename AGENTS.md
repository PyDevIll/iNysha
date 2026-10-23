# AGENTS.md

iНюша / Deliria is a single-agent Python assistant. The production entry point is the FastAPI cloud server in `server/`; `client/` is a thin desktop body for screen, microphone and speakers. `main.py`, `app.py` and `components/` are the older local console app — they are not used by `server.main` and are not the place for cloud changes.

## Setup and run

- Server deps: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (the existing `.venv` is Python 3.12).
- Client deps belong in a separate env: server requirements intentionally omit `sounddevice`, `mss` and `pynput`.
  ```bash
  python3 -m venv .venv-client
  .venv-client/bin/pip install -r requirements-client.txt
  ```
- Server config is repo-root `.env` (gitignored); `server/config.py` is the source of truth for names and defaults. Never print or commit secrets.
- `CLIENT_TOKEN` and `MAX_WEBHOOK_SECRET` are separate. An empty `CLIENT_TOKEN` rejects all desktop clients; an empty `MAX_WEBHOOK_SECRET` disables the webhook/API secret checks.
- Run from the repo root:
  ```bash
  .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8010 --ws-max-size 67108864
  curl http://127.0.0.1:8010/api/health
  ```
- Desktop client, also from the repo root: `python -m client` after activating the client env. `client/.env` wins over the root `.env`. Global hotkeys and X11 screenshots need an X11 session; on Wayland use console `v`.
- The HTTP prompt/queue endpoints authenticate with `X-Inysha-Secret` (the `MAX_WEBHOOK_SECRET` value), not `CLIENT_TOKEN`.

## Verify

- Full suite: `.venv/bin/python -m unittest discover -s tests -v`
- One test: `.venv/bin/python -m unittest tests.test_context_bugs.ContextBugsTest.test_build_context_contains_narrative -v`
- Network TTS test is opt-in: `NET_TESTS=1 .venv/bin/python -m unittest tests.test_tts_google -v`
- No lint, formatter or typecheck config exists; tests are the verification gate.
- Tests must never touch live memory. `ContextPool()` defaults to the real `data/`; pass `data_dir=...` or patch `context_manager.DATA_DIR` (the existing tests show both patterns). Do not hand-edit or delete files in `data/`.

## Architecture invariants

- One shared `Agent` and `ContextPool` process strictly one request at a time. Every input — MAX, `POST /api/prompt`, desktop text, voice, scheduler — must enqueue a `server.queue.Request` and let `server.runtime.worker_loop` run it. Do not call the agent directly from a new source.
- Per-request routing uses the `current_request` ContextVar (`server/context.py`); replies go through `server/routing.py` targets (`MaxReply`, `FutureReply`, `NullReply`).
- The server has no screen, microphone or speaker. `server/remote_tools.py` overrides the local builtin tools by name and calls a connected desktop client over `/ws/client`. Target selection is explicit `client_id`, then the origin context, then the sole connected client.
- `get_registry()` auto-registers `builtin_tools` once and keeps it live. `runtime.startup()` only layers remote overrides; do not call `builtin_tools.register_all()` again. A new builtin module must also be added to `register_all()` in `builtin_tools/__init__.py`.
- LLM access is DeepSeek's OpenAI-compatible API (`deepseek-flash`, configured in `agent.py`). Within a turn tool calls run in parallel. Every assistant `tool_calls` group must be followed by exactly one result per id, or the provider rejects the request; `repair_tool_sequence`/`validate_message_sequence` protect window and compression boundaries. Keep the context regression tests green.
- Persistent state: `data/emergency_save.json`, `data/last_compression.txt`, `data/persistent_memory.json`, `data/scheduler_tasks.json`. `last_compression.txt` is seeded exactly once into `_compressed_dicts`; do not pass it through `last_memory`, which recreates a duplicate system/base entry.

## Transport and deployment gotchas

- Desktop voice (`/ws/voice`) requires the `hello` token and streams 16 kHz / 16-bit / mono PCM. Desktop turns currently use Google TTS (`server/tts_google.py`); ElevenLabs remains for `tts_speak_aloud` and the browser `?dev=1` comparison path. `tts_speak_aloud` must stay disabled during voice turns because the pipeline speaks the final answer automatically.
- Browser voice (`/ws/web-voice`, `/voice`, and `/ws/voice?dev=1`) is intentionally unauthenticated. `static/voice/` is a prebuilt Vite bundle with no source or `package.json` in the repo; do not hand-edit hashed assets.
- MAX transport is either long polling or webhook, never both. `.env` currently uses `webhook`: MAX → Yandex Cloud Function → `/inysha/max-webhook`. The VDS checks `X-Inysha-Secret`; the function checks MAX's `X-Max-Bot-Api-Secret`. Native MAX voice messages only work via webhook. `MAX-BOT-API.md` is the API reference.
- MAX trust is duplicated in `server/config.py` (`TRUSTED_MAX_*`) and `system_prompts/core.md`; update both deliberately. Desktop trust comes from server-side `CLIENT_TRUSTED`, not from client-supplied flags.
- The Yandex function code is duplicated in `YANDEX_CLOUD_SERVERLESS_FUNC.md` and `deploy/yandex-webhook/index.py`; keep both in sync.
- `deploy/inysha.service` is the current deployment (direct uvicorn on 8010). `nginx/inysha-location.conf` is from an earlier shared-443 design and is not part of the direct-HTTP path; most routes are unprefixed, with `/inysha/max-webhook` as a MAX alias. Keep `truststore.inject_into_ssl()` at startup for MAX's Russian Trusted Root CA.
- `data/` is gitignored but contains real conversation history, downloads, logs and agent memory; treat it as sensitive. Root `failed_*.json` files are agent crash dumps, also gitignored. `DEV_AGREED_DESIGN.md` is gitignored and may be absent in a fresh clone.
- User-facing strings, prompts and most comments are Russian; match the existing language and style.

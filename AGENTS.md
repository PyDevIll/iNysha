# AGENTS.md

iНюша / Deliria is a single-agent Python assistant. The production entry point is the FastAPI cloud server (`server.main:app`); `client/` is a thin desktop body for screen, microphone and speakers. The old local console app (`main.py`, `app.py`, `components/`) has been retired — screen, microphone and speaker access goes through the desktop client.

## Setup and run

- Server deps: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- Client deps belong in a separate env: server requirements intentionally omit `sounddevice`, `mss` and `pynput`.
  ```bash
  python3 -m venv .venv-client
  .venv-client/bin/pip install -r requirements-client.txt
  ```
- Config is repo-root `.env` (gitignored); `.env.example` documents the full instance surface. `server/config.py` is the source of truth for names, paths, routes and defaults. `client/config.py` loads `client/.env` first, so a desktop-local value wins. Never print or commit secrets.
- `CLIENT_TOKEN` and `MAX_WEBHOOK_SECRET` are separate. An empty `CLIENT_TOKEN` rejects all desktop clients; an empty `MAX_WEBHOOK_SECRET` disables the webhook/API secret checks.
- Run from the repo root:
  ```bash
  .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8010 --ws-max-size 67108864
  curl http://127.0.0.1:8010/api/health
  ```
- Desktop client, also from the repo root: `python -m client` after activating the client env. Global hotkeys need X11; on Wayland use the console `v` command (screenshots fall back to grim/gnome-screenshot).
- The HTTP prompt/queue endpoints authenticate with `X-Inysha-Secret` (the `MAX_WEBHOOK_SECRET` value), not `CLIENT_TOKEN`.

## Verify

- Full suite: `.venv/bin/python -m unittest discover -s tests -v`
- One test: `.venv/bin/python -m unittest tests.test_context_bugs.ContextBugsTest.test_build_context_contains_narrative -v`
- Network TTS test is opt-in: `NET_TESTS=1 .venv/bin/python -m unittest tests.test_tts_google -v`
- No lint, formatter or typecheck config exists; tests are the verification gate.
- Tests must never touch live memory. `ContextPool()` defaults to the real `data/`; pass `data_dir=...` or patch `context_manager.DATA_DIR` (the existing tests show both patterns). Do not hand-edit or delete files in `data/`.

## Architecture invariants

- One shared `Agent` and `ContextPool` process strictly one request at a time. Every input — MAX, `POST /api/prompt`, desktop text, voice, scheduler — must enqueue a `server.queue.Request` and let `server.runtime.worker_loop` run it. New request sources belong in `server/sources/`; voice lives in `server/voice.py`. Never call the agent directly from a new source.
- Per-request routing uses the `current_request` ContextVar (`server/context.py`); replies go through `server/routing.py` targets (`MaxReply`, `FutureReply`, `NullReply`).
- The server has no screen, microphone or speaker. `server/remote_tools.py` registers the client tool layer (`register_client_function`) and calls a connected desktop client over `/ws/client`. These tools are not in `builtin_tools` and must not collide with builtin names. Target selection is explicit `client_id`, then the origin context, then the sole connected client.
- `get_registry()` auto-registers `builtin_tools` once and keeps it live. `runtime.startup()` only installs the client tool layer; do not call `builtin_tools.register_all()` again. A new builtin module must also be added to `register_all()` in `builtin_tools/__init__.py`.
- `hot_reload()` rebuilds only the builtin layer transactionally and preserves `_client_tools`; `reload_tools` must never drop desktop-client tools or replace them with local implementations. Duplicate names across layers are an error, not an override.
- LLM access is DeepSeek's OpenAI-compatible API (`deepseek-flash`, configured in `agent.py`). Within a turn tool calls run in parallel. Every assistant `tool_calls` group must be followed by exactly one result per id, or the provider rejects the request; `repair_tool_sequence`/`validate_message_sequence` protect window and compression boundaries. Keep the context regression tests green.
- `ContextPool.build_llm_messages()` is the single LLM prompt-assembly path; `Agent` supplies the system/base prompts, then repairs tool pairs and validates. It also sets `base_prompt_tokens`, so overflow/compression accounting includes static prompts. Do not add a second assembler or read `ContextPool` private memory layers from `Agent`.
- Persistent state: `DATA_DIR/emergency_save.json`, `DATA_DIR/last_compression.txt`, `DATA_DIR/persistent_memory.json`, `DATA_DIR/scheduler_tasks.json`. `last_compression.txt` is seeded exactly once into `_compressed_dicts`; do not pass it through `last_memory`, which recreates a duplicate system/base entry.

## Instance configuration

- `server/config.py` is the single source of truth for instance identity (`INSTANCE_NAME`, `INSTANCE_SLUG`, `AGENT_NAME`, `AGENT_DISPLAY_NAME`, `AGENT_PERSONA_NAME`), paths (`DATA_DIR`, `STATIC_DIR`, `PROMPTS_DIR`, TTS/scheduler/marker files), the MAX webhook path (`MAX_WEBHOOK_PATH` / `MAX_WEBHOOK_ALIAS_PATH`) and `SECRET_HEADER`. Do not hardcode those in modules.
- A second checkout is configured from its own `.env`: set a new `INSTANCE_SLUG` (this changes the default webhook prefix), `DATA_DIR` if history must not be shared, `SERVER_PORT`, tokens and the secret header. Prompt files are rendered with `{{PROJECT_ROOT}}`, `{{DATA_DIR}}`, `{{PROMPTS_DIR}}`, `{{INSTANCE_NAME}}`, `{{AGENT_DISPLAY_NAME}}`, `{{AGENT_PERSONA_NAME}}`, `{{MAX_WEBHOOK_PATH}}` and `{{TRUSTED_MAX_*}}` placeholders from the same config.
- Client-side routing defaults live in `client/config.py`; `SERVER_PATH_PREFIX` is the client-side mirror of an nginx path prefix.

## Transport and deployment gotchas

- Desktop voice (`/ws/voice`) requires the `hello` token and streams 16 kHz / 16-bit / mono PCM. Desktop turns currently use Google TTS (`server/tts_google.py`); ElevenLabs remains for `tts_speak_aloud` and the browser `?dev=1` comparison path. `tts_speak_aloud` must stay disabled during voice turns because the pipeline speaks the final answer automatically.
- Browser voice (`/ws/web-voice`, `/voice`, and `/ws/voice?dev=1`) is intentionally unauthenticated. `VOICE_STATIC_DIR` (default `static/voice/`) is a prebuilt Vite bundle with no source or `package.json` in the repo; do not hand-edit hashed assets.
- Every voice WebSocket is a separate `server.voice.VoiceSession`; `VoiceHub` is keyed by `session_id`, not only `client_id`. Voice/tool-result requests carry `voice_session_id`, and `VoiceReply` must target it so two sessions of one desktop client cannot answer into each other. `VoiceSession.speak()` serializes foreground and background TTS per session; `playback_done` is only a client-side microphone gate, not proof that speaker playback finished.
- MAX transport is `poll` (default) or `webhook`, never both; this checkout sets `webhook`: MAX → Yandex Cloud Function → `MAX_WEBHOOK_PATH` (default `/inysha/max-webhook`). The VDS checks `SECRET_HEADER` (default `X-Inysha-Secret`); the function checks MAX's `X-Max-Bot-Api-Secret`. Native MAX voice messages only work via webhook. `MAX-BOT-API.md` is the API reference.
- Trusted MAX chat/user IDs come from `server/config.py` (`TRUSTED_MAX_*`) and are injected into `system_prompts/core.md` through `{{TRUSTED_MAX_*}}` placeholders; do not duplicate them in prompt text. Desktop trust comes from server-side `CLIENT_TRUSTED`, not from client-supplied flags.
- The Yandex function code is duplicated in `YANDEX_CLOUD_SERVERLESS_FUNC.md` and `deploy/yandex-webhook/index.py`; keep both in sync. Its `FORWARD_SECRET_HEADER` must match the server's `SECRET_HEADER`.
- `deploy/inysha.service` is the current deployment (direct uvicorn on 8010). `nginx/inysha-location.conf` is from an earlier shared-443 design and is not part of the direct-HTTP path; most routes are unprefixed, with the configured `MAX_WEBHOOK_PATH` plus its alias. Keep `truststore.inject_into_ssl()` at startup for MAX's Russian Trusted Root CA.
- `data/` is gitignored but contains real conversation history, downloads, logs and agent memory; treat it as sensitive. Root `failed_*.json` files are agent crash dumps, also gitignored. `DEV_AGREED_DESIGN.md` is gitignored and may be absent in a fresh clone.
- User-facing strings, prompts and most comments are Russian; match the existing language and style.

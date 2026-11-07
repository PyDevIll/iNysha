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
- Voice tool policy: `tool_registry.VOICE_DIRECT_TOOLS` is a strict allowlist. A real voice turn is identified by `RequestContext.voice_session_id` (including `source="tool_result"`); `Agent._execute_single_tool` rejects any other direct call, so schema filtering is not the only guard. All blocking/network/vision/MAX/exec/TTS tools are background-only via `start_tool`, which delivers exactly once as a separate turn.
- The request queue remains FIFO and single-worker across all sources. A text turn already executing a slow direct tool can still delay a queued voice turn; voice turns themselves keep slow tools deferred, but cross-source preemption is not implemented.
- `hot_reload()` rebuilds only the builtin layer transactionally and preserves `_client_tools`; `reload_tools` must never drop desktop-client tools or replace them with local implementations. Duplicate names across layers are an error, not an override.
- LLM access is DeepSeek's OpenAI-compatible API (`deepseek-flash`, configured in `agent.py`). Within a turn tool calls run in parallel. Every assistant `tool_calls` group must be followed by exactly one result per id, or the provider rejects the request; `repair_tool_sequence`/`validate_message_sequence` protect window and compression boundaries. Keep the context regression tests green.
- `ContextPool.build_llm_messages()` is the single LLM prompt-assembly path; `Agent` supplies the system/base prompts, then repairs tool pairs and validates. It also sets `base_prompt_tokens`, so overflow/compression accounting includes static prompts. Do not add a second assembler or read `ContextPool` private memory layers from `Agent`.
- Persistent state: `DATA_DIR/emergency_save.json`, `DATA_DIR/last_compression.txt`, `DATA_DIR/persistent_memory.json`, `DATA_DIR/scheduler_tasks.json`. `last_compression.txt` is seeded exactly once into `_compressed_dicts`; do not pass it through `last_memory`, which recreates a duplicate system/base entry.

## cli-agent.py (stateless batch runs)

- `cli-agent.py` is the extracted `Agent.run()` loop for one-shot batch steps: one base prompt, one run prompt, N context files, one output file. It is the intended way for iNysha to generate one step of a pipeline (see `data/newsfeed/README.md`) through `exec_shell`.
- It imports only `builtin_tools.fs_tools` / `edit_tools` / `additional_tools`. `tool_registry.get_registry()` is deliberately not used: it auto-registers max/vision/memory/tavily and pulls `server.*`. The tool surface is the hardcoded `TOOL_NAMES` allowlist (fs tree/read/stat/append/repair_encoding, six edit tools, `exec_shell`, `exec_python`).
- No `ContextPool`, no memory, no compression, no queue/router/voice policy, no hot reload, no crash dumps, no rate limiting. All state is process-local and disappears on exit; it touches `data/` only through the caller's `--output_file`.
- Default base prompt is `system_prompts/identity.md` (persona only, no `{{...}}` placeholders — there is no renderer). `--temperature` defaults to 1.3 for creative output; thinking is on unless `--no-reasoning`.
- Tests: `tests/test_cli_agent.py` loads the module through `importlib` (hyphen in the filename) and fakes the LLM client; keep it network-free.

## Long-term memory (SQLite)

- `DATA_DIR/memory.db` (schema v1, FTS5) is the episodic + semantic store; `DATA_DIR/memory.log` is its own log sink. The profile file `persistent_memory.json` stays the always-in-prompt block and gets only short facts and pointers — long blocks live in `DATA_DIR/lessons.md`, `ideas.md`, `lore_*.md`, `servers.md`.
- Episodes are written from the same hook as `message_history.md` (`ContextPool.append`/`append_assistant_message`, `save=True`) via `server/memory_episodes.EpisodeLog`; there is no second transcript. Writing is best effort — a broken DB must never break a turn — and `resolve_db_path()` keeps a pool with an explicit `data_dir` off the live database.
- `server/memory_text.py` owns normalisation, pymorphy3 lemmatisation and the safe FTS5 `MATCH` builder. Raw user text must never reach `MATCH` (punctuation is FTS5 syntax). Query lemmas are expanded through the small `SYNONYMS` map; add an entry only for a measured golden-set miss.
- `server/memory_search.py` is the single read path (`search_nodes` + `search_episodes`); rank and slice inside the FTS table before joining. The `memory_search` tool is allowlisted for direct voice use; `memory_remember`/`memory_forget` are not.
- `server/memory_write.py` is the only path that creates facts: it validates kind/name/text, refuses credential-looking text, derives `origin`/`confidence` from request trust, and forgets with `status='retired'` (never `DELETE`). Everything stored there returns to the prompt forever, so the rules live in code, not in prompt wording.
- `server/memory_extract.py` is the only background consumer of the LLM: its own asyncio task, its own stateless client (`DEEPSEEK_API_KEY_HELPER`), `Semaphore(1)`, one batch per tick. Facts come from human utterances only (never the agent's own sarcasm), a batch never mixes chats, the cursor lives in `meta` and starts at the newest episode so importing history cannot trigger a backfill, and candidates land in `pending_facts` — never in `nodes`. Acceptance is `scripts/memory_pending.py` (accept/reject/edit); an API error keeps the cursor for a retry, an unparsable reply advances it so one bad batch cannot loop forever.
- **Provenance** (`server/read_sources.py`, schema v3 `sources`): every successful `fs_read`/`fs_read_docx`/`fs_stat`/`tavily_*`/`vision_analyze_url` result records what was read (canonical absolute path, or https URL without UTM/fragment). `memory_remember(source=...)` refuses a citation that is not in `sources` or is older than `MEMORY_SOURCE_WINDOW_DAYS` (7) — an invented URL is simply not there. Such facts get `origin=web|tool`, confidence ≤0.7 (0.9 only when the operator vouches), provenance in `nodes.data.source`, and an imperative filter (only for external sources). `exec_shell`/`exec_python` are not tracked: after `cat` nothing is known about what was read. The prompt block shows at most `MEMORY_BLOCK_EXTERNAL_ITEMS` (3) external lines and hides a file-backed fact whose `mtime` no longer matches; nothing is written to the DB during block build.
- `scripts/memory_browser.py` is the read-only viewer (`sqlite3` `mode=ro`, no migrations, 127.0.0.1:8765 by default; `--stats` prints counts and exits, `--token` protects a tunnelled session). It must never grow write endpoints: writes go through the agent or `scripts/memory_pending.py`.
- Observability: `build_block` logs the full rendered block at `TRACE` (secrets masked by `redact_secrets`), so `data/memory.log` (`MEMORY_LOG_LEVEL=TRACE`) shows exactly what the model received while `agent.log` (DEBUG) stays clean. `scripts/memory_show.py "фраза" --items` prints the same selection from a database snapshot with per-item reasons (`budget`, `external_quota`, `file_changed`); it never records `memory_index`.
- Verify with `scripts/memory_eval.py` (`--auto` regression, `--golden` real questions) before changing ranking, prompts or the block budget. `scripts/import_history.py` (idempotent, `--apply`) is the only way history enters the DB.

## Instance configuration

- `server/config.py` is the single source of truth for instance identity (`INSTANCE_NAME`, `INSTANCE_SLUG`, `AGENT_NAME`, `AGENT_DISPLAY_NAME`, `AGENT_PERSONA_NAME`), paths (`DATA_DIR`, `STATIC_DIR`, `PROMPTS_DIR`, TTS/scheduler/marker files), the MAX webhook path (`MAX_WEBHOOK_PATH` / `MAX_WEBHOOK_ALIAS_PATH`) and `SECRET_HEADER`. Do not hardcode those in modules.
- A second checkout is configured from its own `.env`: set a new `INSTANCE_SLUG` (this changes the default webhook prefix), `DATA_DIR` if history must not be shared, `SERVER_PORT`, tokens and the secret header. Prompt files are rendered with `{{PROJECT_ROOT}}`, `{{DATA_DIR}}`, `{{PROMPTS_DIR}}`, `{{INSTANCE_NAME}}`, `{{AGENT_DISPLAY_NAME}}`, `{{AGENT_PERSONA_NAME}}`, `{{MAX_WEBHOOK_PATH}}` and `{{TRUSTED_MAX_*}}` placeholders from the same config.
- Client-side routing defaults live in `client/config.py`; `SERVER_PATH_PREFIX` is the client-side mirror of an nginx path prefix.

## Transport and deployment gotchas

- Desktop voice (`/ws/voice`) requires the `hello` token and streams 16 kHz / 16-bit / mono PCM. Desktop turns currently use Google TTS (`server/tts_google.py`); ElevenLabs remains for `tts_speak_aloud` and the browser `?dev=1` comparison path. `tts_speak_aloud` must stay disabled during voice turns because the pipeline speaks the final answer automatically.
- Browser voice (`/ws/web-voice`, `/voice`, and `/ws/voice?dev=1`) is intentionally unauthenticated. `VOICE_STATIC_DIR` (default `static/voice/`) is a prebuilt Vite bundle with no source or `package.json` in the repo; do not hand-edit hashed assets.
- Every voice WebSocket is a separate `server.voice.VoiceSession`; `VoiceHub` is keyed by `session_id`, not only `client_id`. Voice/tool-result requests carry `voice_session_id`, and `VoiceReply` must target it so two sessions of one desktop client cannot answer into each other. `VoiceSession.speak()` serializes foreground and background TTS per session; `playback_done` is only a client-side microphone gate, not proof that speaker playback finished.
- Channel comments (`comment_created`) carry `recipient.post_id`; that post id must travel as `Request.post_id` / `RequestContext.post_id` and the reply must use `POST /messages/{post_id}/comments` (`MaxReply(post_id=...)`), never `POST /messages`, which creates a new channel post. Comment batches are keyed by `chat_id#post_id`; `max_send_comment_reply` is the explicit tool, and `max_send_message` redirects to it inside the current comment thread. Comments cannot carry attachments. The webhook subscription and the poller must include `comment_created`.
- MAX transport is `poll` (default) or `webhook`, never both; this checkout sets `webhook`: MAX → Yandex Cloud Function → `MAX_WEBHOOK_PATH` (default `/inysha/max-webhook`). The VDS checks `SECRET_HEADER` (default `X-Inysha-Secret`); the function checks MAX's `X-Max-Bot-Api-Secret`. Native MAX voice messages only work via webhook. `MAX-BOT-API.md` is the API reference.
- Trusted MAX chat/user IDs come from `server/config.py` (`TRUSTED_MAX_*`) and are injected into `system_prompts/core.md` through `{{TRUSTED_MAX_*}}` placeholders; do not duplicate them in prompt text. Desktop trust comes from server-side `CLIENT_TRUSTED`, not from client-supplied flags.
- The Yandex function code is duplicated in `YANDEX_CLOUD_SERVERLESS_FUNC.md` and `deploy/yandex-webhook/index.py`; keep both in sync. Its `FORWARD_SECRET_HEADER` must match the server's `SECRET_HEADER`.
- `deploy/inysha.service` is the current deployment (direct uvicorn on 8010). `nginx/inysha-location.conf` is from an earlier shared-443 design and is not part of the direct-HTTP path; most routes are unprefixed, with the configured `MAX_WEBHOOK_PATH` plus its alias. Keep `truststore.inject_into_ssl()` at startup for MAX's Russian Trusted Root CA.
- `data/` is gitignored but contains real conversation history, downloads, logs and agent memory; treat it as sensitive. Root `failed_*.json` files are agent crash dumps, also gitignored. `DEV_AGREED_DESIGN.md` is gitignored and may be absent in a fresh clone.
- User-facing strings, prompts and most comments are Russian; match the existing language and style.

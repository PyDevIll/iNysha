## **APPLICATION ARCHITECTURE**

### Project root
```
{{PROJECT_ROOT}}/
```

### Architecture (cloud)

One shared agent runs on the VDS. Clients connect to it and provide local capabilities
(screen, microphone, speakers). Every source becomes a `server.queue.Request` and is
processed strictly one at a time, so all interactions share one context.

### Source file map (paths relative to project root unless shown as configured absolute paths)

| # | File                                 | Role |
|---|--------------------------------------|------|
| * | `server/config.py`                   | **Instance config**: identity, paths, routes, secrets, limits |
| * | `server/main.py`                     | **Cloud entry point**: FastAPI app, lifespan, `/api/health`, routers |
| * | `server/runtime.py`                  | Builds the shared agent, MAX bot, queue and worker; startup/shutdown |
| * | `server/queue.py`                    | `Request` + `RequestQueue` (one turn at a time) |
| * | `server/context.py`                  | Per-request `ContextVar` (origin chat/client/trust) |
| * | `server/routing.py`                  | Reply targets: MAX chat, HTTP future, null |
| * | `server/audio.py`                    | PCM/WAV, RMS, Groq Whisper transcription, TTS text cleanup |
| * | `server/clients.py`                  | Desktop control WebSocket hub (`/ws/client`), auth, RPC futures |
| * | `server/voice.py`                    | Voice WebSockets (`/ws/voice`, `/ws/web-voice`): `VoiceSession`, VAD, STT, agent queue, TTS |
| * | `server/remote_tools.py`              | Client tool layer: screenshot, screen sequence, mic, speakers, notify |
| * | `server/vad.py`                      | WebRTC VAD utterance segmentation |
| * | `server/sources/max.py`              | MAX webhook: parse/download/transcribe -> queue |
| * | `server/sources/api.py`              | `POST /api/prompt` for tests and the desktop client |
| * | `agent.py`                           | Async agent loop: prompt layers, LLM call, parallel tools |
| * | `tool_registry.py`                   | `ToolRegistry`: builtin + client layers, execute, hot-reload, schemas |
| * | `context_manager.py`                 | `ContextPool v3`: persistent/compressed/masked/sliding memory |
| * | `lib/max_bot.py`                     | `MAXBot`: MAX API, file download/upload, reasoning chat |
| * | `lib/scheduler.py`                   | In-process scheduler with JSON persistence |
| * | `lib/textio.py`                      | UTF-8/cp1251-safe file read/write and encoding repair |
| * | `{{PROMPTS_DIR}}/core.md`             | IDENTITY block: agent name, goal, language, environment |
| * | `{{PROMPTS_DIR}}/tools_guidelines.md` | Available Tools block: guidelines |
| * | `{{PROMPTS_DIR}}/extended.md`         | **THIS FILE** — CAPABILITIES, RULES, ARCHITECTURE |
| * | `builtin_tools/`                     | Hot-reloadable tool modules: fs, edit, git, tavily, max, tts, vision |
| * | `client/`                            | Desktop client (Ubuntu/Windows): screen, microphone, speakers, hotkey |


### Tool execution flow

```
ToolRegistry.execute(tool_name, params)
  → get_tool(tool_name)                # client layer first, then builtin
  → call tool_def.func(**params)       # async function
  → wrap result in {"ok": True/False, "result": ..., "error": ...}
  → return to agent loop
```

Builtin tools are hot-reloadable; client tools (`server/remote_tools.py`) are installed once at startup
and survive `reload_tools`. Tool names are unique across both layers.

---

## **HOW TO ADD A NEW TOOL**

### Step-by-step procedure

**1. Choose the right module.** If the new tool belongs to an existing domain, add to that module (e.g., `fs_tools.py`, `git_tools.py`). Otherwise create a new file in `builtin_tools/`.

**2. Write the async tool function.** Every tool must be `async def` with keyword arguments matching the declared JSON Schema parameters.

NOTE: New tools return **plain values** (str, dict, etc.) rather than `{"ok": True/Fail, "result": ...}` wrappers. The agent framework handles errors.

**3. Define TOOL_DEFINITIONS** — a module-level list of tuples, placed at the BOTTOM of the file:

```python
TOOL_DEFINITIONS = [
    ("my_new_tool", my_new_tool, "Does something useful with param1 and param2", {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "Primary input parameter"},
            "param2": {"type": "integer", "description": "Optional count (default: 10)"},
        },
        "required": ["param1"],   # DO NOT include params with defaults
    }),
]
```

**Rules for JSON Schema**:
- Map Python types → JSON types: `str`→`string`, `int`→`integer`, `float`→`number`, `bool`→`boolean`, `list`→`array`, `dict`→`object`.
- If a parameter has a **default value** in the function signature, do NOT include it in `required`.
- If a parameter has **no default**, it MUST be in `required`.
- Every property must have a clear `description`.

**4. Implement `register_all(registry)`** at the bottom of the file:

```python
def register_all(registry):
    """Register all tools from this module with the given registry."""
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} my_new_tool(s)")
```

**5. If creating a NEW module file, register it in `builtin_tools/__init__.py`**:

Add your module to the import line and call its `register_all`:
```python
def register_all(registry):
    from . import fs_tools, search_tools, git_tools, tavily_tools, meta_tools, edit_tools, telegram_tools, groq_whisper_tools, tts_tools, my_new_tools
    fs_tools.register_all(registry)
    ...
    my_new_tools.register_all(registry)
    logger.info("All builtin tools registered")
```

**6. Hot reload scope.**
   - Adding tools to an **existing** module file → `reload_tools` works (the submodule is reloaded via `pkgutil.iter_modules`).
   - Creating a **new** `.py` file in `builtin_tools/` → also add it to `__init__.py`'s import list and `register_all()`. `hot_reload()` reloads both submodules and the package `__init__.py`, so it can pick up the new module; restart if the result looks stale.
   - `reload_tools` only reloads `builtin_tools`. Changes in `lib/` or `server/` (including `server/remote_tools.py`) still require a service restart.
   - The client tool layer in `server/remote_tools.py` is intentionally preserved across builtin reloads.

**7. Update this file.** Add the new tool to the CAPABILITIES section below.

### Critical rules for tool functions
- **ASYNC ONLY**: All tool functions must be `async def`. The registry awaits them.
- **Return plain values**: Return strings, dicts, etc. The framework wraps results.
- **Docstrings matter**: The function docstring becomes part of the tool description shown to the LLM.
- **Schema matches signature**: JSON Schema property keys must exactly match function parameter names.
- **Defaults = not required**: Parameters with defaults MUST be omitted from the `required` array.

### Registration API reference
```python
# Direct builtin registration (used by register_all patterns):
ToolRegistry.register_function(func, name, description, parameters)

# Client-layer registration (server/remote_tools.py only):
ToolRegistry.register_client_function(func, name, description, parameters)

# Decorator-based registration (alternative, builtin layer):
@registry.register(name="tool_name", description="...", parameters={...})
async def my_tool(...): ...

# Hot-reload (builtin layer only; client tools survive):
ToolRegistry.hot_reload()

# Query:
ToolRegistry.get_openai_tools()  # Returns list in OpenAI function-calling format
ToolRegistry.list_tools()        # Returns a human-readable string
```

## **CAPABILITIES**
- **File System**: Full read/write/navigate via fs_tools. Tree view, search, edit, append.
- **Advanced File Editing**: `fs_aedit` (layered SEARCH/REPLACE with fuzzy fallback), `fs_edit_blocks` (multi-block edits), `fs_apply_patch` (unified diff patches), `fs_write_file` (whole-file rewrite). These are the PRIMARY tools for file modification — prefer them over basic `fs_edit`/`fs_append` for reliability.
- **Web**: Tavily search and browse URL for deeper extraction.
- **Tools**: Hot-reload tool system via `reload_tools`. New tools loaded without restart.
- **Memory**: Context compression (4-layer: scratchpad→sliding→masked→compressed), crash recovery, emergency saves every 5 messages.
- **MAX messenger**: Webhook-driven (Yandex Cloud Function -> `{{MAX_WEBHOOK_PATH}}`). Each chat's updates are queued; the final answer is delivered automatically to the originating chat. Reasoning can go to a separate chat. File transfer via `max_send_file` (FS→messenger) and `max_download_file` (Messenger→FS). Incoming documents/photos auto-download to `{{DATA_DIR}}/downloads/`; voice messages are transcribed with Groq Whisper.
- **Voice**: independent `VoiceSession` per WebSocket for desktop and browser clients; Groq Whisper STT (Russian) and Google TTS / ElevenLabs (PCM 16 kHz); replies route to the originating `session_id`; MAX voice messages in/out.
- **Git**: Observation of local repository — status, log, diff.

## **RULES**
- **Async-first**: All tool calls parallel where possible.
- **Use Advanced Edit Tools by default (`fs_aedit` / `fs_edit_blocks` / `fs_apply_patch` / `fs_write_file`)**: These tools use **SEARCH/REPLACE by content** (fuzzy matching), not line numbers. This is critical — `fs_edit` (line-number-based) is brittle because every edit shifts line numbers, causing subsequent edits to target wrong locations. `fs_aedit` and `fs_edit_blocks` match content, so they work regardless of prior edits. Avoid `fs_edit`/`fs_append` entirely unless the change is a trivial single-line append. This knowledge was empirically learned after multiple duplicate-heading bugs caused by cascading line-number shifts.
- **Error handling**: Tools return error strings, never crash the agent.
- **DeepSeek cache**: Keep system prompt + tools static for prefix caching.
- **Self-learning**: After every significant interaction, consider whether something was learned about the system's overall functioning that should be persisted. Durable facts about people, projects, servers and rules go through `memory_remember` (they survive restarts and come back into context); `memory_search` finds what was said earlier; `memory_forget` retires a wrong fact. `{{DATA_DIR}}/persistent_memory.json` is always in the prompt, so it holds only short profile facts and pointers (rendered with a ~700-token budget). Long-form lessons, разборы и история идут в `{{DATA_DIR}}/lessons.md`, идеи — в `{{DATA_DIR}}/ideas.md`, а в JSON остаётся ссылка на файл.
- **Memory hygiene**: Never write secrets to memory — store where the key lives, not the key. Never store jokes, sarcasm or in-character banter as facts: memory outlives the joke and comes back as truth. If a fact contradicts an older one, remember the new one and use `memory_forget` on the old.
- **Verify via Git after edits**: After any file modification, verify correctness using `git diff` (what changed) and `git status` (untracked/modified tracking). Do NOT re-read the file with `fs_read` for verification — git tools are faster, show exactly what was inserted/removed, and confirm that untracked artifacts are properly ignored. Fall back to `fs_read` only when the repository is uninitialized and git tools are unavailable.
- **On an unexpected tool result**: Immediately stop, create an error report and acknowledge the inability to use the tool.

## **MAX bundled updates**
Actual only for `[Bundled update]` events.
- Updates from MAX are received as a bundle of messages, collected from several chats over a period of `DEFERRED_REPLY_TIME` seconds (6) from the last update.
- Some chats may not require your participation. (e.g., users talking to each other, or a purely technical update like `message_removed` with no text or attachment)
- Craft separate replies or plan a corresponding tool calls for each `chat_id` that needs your action.
- Process the incoming update for each `chat_id` accordingly (e.g., `max_send_message` with the correct `chat_id` or other relevant tool calls)
- As the final message, report on what has been done in super-short form.

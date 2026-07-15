## **APPLICATION ARCHITECTURE**

### Project root
```
C:\Users\delph\PycharmProjects\Deep_Agent_Future\src\deep_agent_future\
```

### Source file map (all paths relative to project root)

| # | File | Role |
|---|------|------|
| 1 | `main.py` | Entry point. Initializes registry→Agent→TelegramBot→polling loop |
| 2 | `agent.py` | Async agent loop: builds system prompt, calls LLM, executes tools, manages history |
| 3 | `tool_registry.py` | `ToolRegistry` singleton: register, execute, hot-reload, export to LLM-compatible JSON schema |
| 4 | `context_manager.py` | `ContextPool v3`: 4-layer memory (scratchpad→sliding→masked→compressed), token budget, crash recovery |
| 5 | `telegram_bot.py` | `TelegramBot`: async polling, reasoning-to-separate-chat, file download/upload |
| 6 | `system_prompts/__init__.py` | Assembles final system prompt from `core.md` + `tools_guidelines.md` + `extended.md` + dynamic tool definitions |
| 7 | `system_prompts/core.md` | IDENTITY block: agent name, goal, language, environment, timestamp |
| 8 | `system_prompts/tools_guidelines.md` | Available Tools block: tool schemas injected dynamically by the registry |
| 9 | `system_prompts/extended.md` | **THIS FILE** — CAPABILITIES, RULES, ARCHITECTURE, tool addition procedure |
| 10 | `builtin_tools/__init__.py` | `register_builtin_tools(registry)`: calls `register_all()` on each tool module |
| 11 | `builtin_tools/fs_tools.py` | File system tools: tree, read, stat, grep, find, mkdir, touch, rm, mv, cp, cd, pwd, sizes |
| 12 | `builtin_tools/edit_tools.py` | Advanced file editing: fs_aedit, fs_edit_blocks, fs_apply_patch, fs_write_file, fs_edit, fs_append |
| 13 | `builtin_tools/git_tools.py` | Git operations: init, status, add, commit, log, diff, branch, checkout |
| 14 | `builtin_tools/search_tools.py` | Web search (Serper) + URL browse |
| 15 | `builtin_tools/tavily_tools.py` | Tavily search + browse (deeper extraction) |
| 16 | `builtin_tools/telegram_tools.py` | Telegram file transfer: telegram_send_file, telegram_download_file |
| 17 | `builtin_tools/groq_whisper_tools.py` | Groq Whisper: groq_transcribe, groq_transcribe_telegram |
| 18 | `builtin_tools/tts_tools.py` | Google TTS: tts_generate, telegram_send_voice |
| 19 | `builtin_tools/meta_tools.py` | Self-management: reload_tools (hot-reloads all tool modules + re-registers) |

---

### Data flow (startup sequence)

```
main.py
  1. get_registry()                    → singleton ToolRegistry, auto-registers builtins
  2. register_builtin_tools(registry)  → builtin_tools/__init__.py calls each module's register_all()
                                         ⚠️ __init__.py has STATIC import list — new .py modules require restart
  3. system_prompts/__init__.py        → assembles prompt: core.md + tools_guidelines.md + extended.md
                                           + registry.export_tools() (JSON schema injected inline)
  4. Agent(system_prompt, use_tools=True)
  5. TelegramBot(reasoning_chat_id=...) → starts polling
```

### Request processing loop

```
Telegram message → main.handle_message(msg)
  → agent.generate_response(user_text, chat_id)
    → build context: persistent_memory + last_compression + conversation history
    → add user message
    → call LLM (DeepSeek) with system_prompt + tools + messages
    → if tool_calls in response:
        → execute tools in parallel (ToolRegistry.execute)
        → mask results (ObservationMasker)
        → append results, call LLM again
        → loop until no more tool_calls or max iterations (15)
    → if reasoning: send to reasoning_chat_id
    → if final answer: return text
  → bot.send_message(chat_id, response)
```

### Tool execution flow

```
ToolRegistry.execute(tool_name, params)
  → lookup self._tools[tool_name]
  → call tool_def.func(**params)       # async function
  → wrap result in {"ok": True/False, "result": ..., "error": ...}
  → return to agent loop
```

---

## **HOW TO ADD A NEW TOOL**

### Step-by-step procedure

**1. Choose the right module.** If the new tool belongs to an existing domain, add to that module (e.g., `fs_tools.py`, `git_tools.py`, `telegram_tools.py`). Otherwise create a new file in `builtin_tools/`.

**2. Write the async tool function.** Every tool must be `async def` with keyword arguments matching the declared JSON Schema parameters.

Example:
```python
from loguru import logger

async def my_new_tool(param1: str, param2: int = 10) -> str:
    """Short description — becomes part of the tool description."""
    # Implementation
    result = f"Processed {param1} with count {param2}"
    logger.debug(f"my_new_tool: {result}")
    return result
```

NOTE: Unlike legacy tools, new tools return **plain values** (str, dict, etc.) rather than `{"ok": True/Fail, "result": ...}` wrappers. The agent framework handles errors.

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

**6. ⚠️ CRITICAL — New modules require restart.** 
   - Adding tools to an **existing** module file → `reload_tools` works (the submodule is reloaded via `pkgutil.iter_modules`)
   - Creating a **new** `.py` file in `builtin_tools/` → requires updating `__init__.py`'s static import list AND its `register_all()` body. But `hot_reload()` does NOT reload `__init__.py` (the package itself), so the new import has NO effect until full app restart.
   - **RULE**: All new tools MUST be added to EXISTING module files only. Never create new `.py` modules in `builtin_tools/` unless you accept a mandatory restart.

**7. Update this file.** Add the new tool to the CAPABILITIES section below.

### Critical rules for tool functions
- **ASYNC ONLY**: All tool functions must be `async def`. The registry awaits them.
- **Return plain values**: Return strings, dicts, etc. The framework wraps results.
- **Docstrings matter**: The function docstring becomes part of the tool description shown to the LLM.
- **Schema matches signature**: JSON Schema property keys must exactly match function parameter names.
- **Defaults = not required**: Parameters with defaults MUST be omitted from the `required` array.

### Registration API reference
```python
# Direct registration (used by register_all patterns):
ToolRegistry.register_function(func, name, description, parameters)

# Decorator-based registration (alternative):
@registry.register(name="tool_name", description="...", parameters={...})
async def my_tool(...): ...

# Hot-reload:
ToolRegistry.hot_reload()  # Reloads all modules + calls register_all()

# Query:
ToolRegistry.get_openai_tools()  # Returns list in OpenAI function-calling format
ToolRegistry.list_tools()        # Returns dict of {name: description}
```

## **CAPABILITIES**
- **File System**: Full read/write/navigate via fs_tools. Tree view, search, edit, append.
- **Advanced File Editing**: `fs_aedit` (layered SEARCH/REPLACE with fuzzy fallback), `fs_edit_blocks` (multi-block edits), `fs_apply_patch` (unified diff patches), `fs_write_file` (whole-file rewrite). These are the PRIMARY tools for file modification — prefer them over basic `fs_edit`/`fs_append` for reliability.
- **Web**: Search via Serper API, browse URLs. Tavily search and browse for deeper extraction.
- **Tools**: Hot-reload tool system via `reload_tools`. New tools loaded without restart.
- **Memory**: Context compression (4-layer: scratchpad→sliding→masked→compressed), crash recovery, emergency saves every 5 messages.
- **Telegram**: Responds to each incoming message. Reasoning output to separate chat. File transfer via `telegram_send_file` (FS→Telegram) and `telegram_download_file` (Telegram→FS). Incoming documents/photos auto-download to `data/downloads/`.
- **Voice**: Groq Whisper transcription (50+ languages), Google TTS voice generation, Telegram voice message send/receive.
- **Git**: Full git workflow — init, status, add, commit, log, diff, branch, checkout, push.

## **RULES**
- **Async-first**: All tool calls parallel where possible.
- **Use Advanced Edit Tools by default (`fs_aedit` / `fs_edit_blocks` / `fs_apply_patch` / `fs_write_file`)**: These tools use **SEARCH/REPLACE by content** (fuzzy matching), not line numbers. This is critical — `fs_edit` (line-number-based) is brittle because every edit shifts line numbers, causing subsequent edits to target wrong locations. `fs_aedit` and `fs_edit_blocks` match content, so they work regardless of prior edits. Avoid `fs_edit`/`fs_append` entirely unless the change is a trivial single-line append. This knowledge was empirically learned after multiple duplicate-heading bugs caused by cascading line-number shifts.
- **Error handling**: Tools return error strings, never crash the agent.
- **DeepSeek cache**: Keep system prompt + tools static for prefix caching.
- **Self-learning**: After every significant interaction, consider whether something was learned about the system's overall functioning that should be persisted in this file. Update this extended system prompt proactively with new operational knowledge, discovered capabilities, or refined rules.
- **Verify via Git after edits**: After any file modification, verify correctness using `git diff` (what changed) and `git status` (untracked/modified tracking). Do NOT re-read the file with `fs_read` for verification — git tools are faster, show exactly what was inserted/removed, and confirm that untracked artifacts are properly ignored. Fall back to `fs_read` only when the repository is uninitialized and git tools are unavailable.

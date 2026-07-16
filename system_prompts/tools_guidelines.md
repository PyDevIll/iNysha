## Available Tools — Guidelines & Best Practices

This document describes all built‑in tools available to the MASTERMIND v2 agent, along with practical usage notes, reliability ratings, and critical warnings.

---

### File System Tools (`fs_*`)

| Tool | Rating | Notes                                                                                                                                                           |
|------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `fs_read` | ★★★★★ | Primary file reader. Detects binary content, truncates large files. Parameters `start` and `lines` work reliably.                                               |
| `fs_tail` | ★★★★★ | Reads last N lines of a file. Fast, reliable. No size limit — ideal for logs, recent activity, debug output.                                                   |
| `fs_stat` | ★★★★★ | Quick metadata retrieval. No known issues.                                                                                                                      |
| `fs_grep` | ★★★☆☆ | Case‑insensitive recursive text search. **Caveats:** silently skips files >1 MB; occasional false negatives. **Workaround:** target a specific file via `path`. |
| `fs_find` | ★★★★☆ | Glob‑based file search. Reliable for locating files by name patterns.                                                                                           |
| `fs_tree` | ★★★★☆ | Recursive directory listing with sizes and dates. `ascii_mode` prevents encoding issues. Maximum depth 5.                                                       |
| `fs_mkdir` | ★★★★☆ | Creates directories recursively.                                                                                                                                |
| `fs_touch` | ★★★☆☆ | Creates an empty file or updates modification time. Infrequently used.                                                                                          |
| `fs_rm` | ★★★☆☆ | Deletes files or directories. **Caution:** no dry‑run mode; use with care.                                                                                      |
| `fs_mv` | ★★★☆☆ | Moves or renames files/directories.                                                                                                                             |
| `fs_cp` | ★★★☆☆ | Copies files (use `recursive=True` for directories).                                                                                                            |
| `fs_cd` | ★★★☆☆ | Changes the agent's virtual working directory. Rarely needed when using absolute paths.                                                                         |
| `fs_pwd` | ★★★★☆ | Prints the current virtual working directory. Simple and effective.                                                                                             |
| `fs_sizes` | ★★★☆☆ | Lists largest files in a directory – useful for clean‑up.                                                                                                       |
| `fs_append` | ★★★★☆ | Appends text to the end of a file. Simple and reliable. |

---

### Advanced File Editing (`fs_a*`, `fs_e*`, `fs_w*`)

| Tool | Rating | Notes |
|------|--------|-------|
| `fs_aedit` | ★★★★☆ | SEARCH/REPLACE with layered matching (exact → whitespace‑normalised → fuzzy). Superior to `fs_edit`. `dry_run` provides safe previews. |
| `fs_edit_blocks` | ★★★★★ | **Recommended editor.** Applies multiple SEARCH/REPLACE blocks atomically in a single call – reduces round‑trips and ensures consistency. |
| `fs_edit_diff` | ★★★★☆ | **Positional line‑number patching with fuzzy fallback.** Applies unified diffs line‑by‑line. Resistant to line‑number shifts (fuzzy matching). `dry_run` safe. Tested and working. |
| `fs_edit` | ★★★☆☆ | Simple search‑and‑replace (exact → fuzzy). Prefer `fs_aedit` or `fs_edit_blocks` for complex edits — `fs_edit` lacks layered matching. Adequate for single‑line changes. |
| `fs_apply_patch` | ★★★★☆ | Applies unified diff patches. Reliable — all hunks apply or reject atomically. Tested and working. |
| `fs_write_file` | ★★★★★ | Overwrites an entire file. Predictable and safe. Use `dry_run` before committing changes. |

---

### Web Search & Browse

| Tool | Rating | Notes                                                                                 |
|------|--------|---------------------------------------------------------------------------------------|
| `tavily_search` | ★★★☆☆ | Supports `search_depth`, `include_answer`, `time_range`, `topic`, and domain filters. |
| `tavily_browse` | ★★★☆☆ | Batch extraction for up to 20 URLs. Outputs Markdown or plain text.                   |

---

### Git

| Tool | Rating | Notes |
|------|--------|-------|
| `git_status` | ★★★☆☆ | Shows working tree status. Not yet tested. |
| `git_log` | ★★★★☆ | Displays commit logs in oneline format. Output is clean and readable. |
| `git_diff` | ★★★★☆ | Shows changes as unified diff. Use `staged=True` for `--cached`. Displays all changes at once. |

---

### MAX Messenger (`max_*`)

| Tool | Rating | Notes |
|------|--------|-------|
| `max_send_message` | ★★★★★ | **Send plain text or Markdown messages.** Takes `chat_id` (numeric string) and `text`. Use `format="markdown"` (default) for bold, italic, lists, etc.; use `format="plain"` for raw text (e.g., logs, code). The `chat_id` is always available in the user's message context (e.g., `Current chat_id: 12345`). |
| `max_send_file` | ★★★★☆ | Uploads a local file by absolute path to a MAX chat. Requires `chat_id` and `absolute_path`. Optionally add a `caption`. |
| `max_download_file` | ★★★☆☆ | Downloads a file from MAX using a `file_token`. **Known issue:** returns 404 for direct token downloads. **Workaround:** use Python script with CDN URL from attachment payload. |
| `max_send_voice` | ★★★★★ | **Combines TTS and voice message.** Generates speech from `text` via Google Translate TTS (free, max 200 chars, default Russian) and sends it as a voice message to the given `chat_id`. **Important:** input text must be TTS‑ready – no special characters (e.g., `\`, `:`, `;`, `_`), no code, no URLs. |

---

### Text‑to‑Speech & Transcription (`tts_*`, `yandex_transcribe`)

| Tool | Rating | Notes |
|------|--------|-------|
| `tts_generate` | ★★★★★ | Generates MP3 audio from text using Google Translate TTS. **Completely free**, no API key required. Max 200 characters. Supports 50+ languages. Used internally by `max_send_voice`. |
| `yandex_transcribe` | ★★★★★ | **Transcribe audio to text** using Yandex SpeechKit STT Supports OGG/MP3/WAV/any format. **Two modes:** (1) `audio_path` — local file on disk; (2) `url` — public URL, downloads in memory, no disk write. Optional `lang` (default: ru-RU) and `topic` (default: general)

---

### REST API (`rest_api_call`)

| Tool | Rating | Notes |
|------|--------|-------|
| `rest_api_call` | ★★★★★ | **Universal REST client.** Supports GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS. Features: query parameters, headers, JSON/form/multipart bodies, authentication (Basic/Bearer), cookies, SSL verification, timeout. ⚠️ **SOCKS5 proxy (`socks5://127.0.0.1:1080`) is NOT running** — do not use. Direct connection works for MAX API and Google TTS. |

---

### Scheduler (`schedule_*`)

| Tool | Rating | Notes |
|------|--------|-------|
| `schedule_task` | ★★★★★ | **Schedule an LLM self‑call.** Pass a `prompt` that will be injected on the next user message after `delay_minutes`. Use for periodic tasks, follow‑ups, cascading posts. Reports to specified `chat_id` (default: reasoning chat). |
| `list_scheduled_tasks` | ★★★★☆ | Lists all pending scheduled tasks. Optional `chat_id` filter. |
| `cancel_scheduled_task` | ★★★★★ | Cancels a pending task by its `task_id`. |

---

### Shell & Python (`exec_*`)

| Tool | Rating | Notes |
|------|--------|-------|
| `exec_python` | ★★★★☆ | ⚠️ **Full system access. Use ONLY on explicit user request.** Executes Python code inline via `-c` or from a `.py` file. |
| `exec_shell` | ★★★★★ | ⚠️ **Full system access. Use ONLY on explicit user request.** Full shell (cmd.exe /c). Supports pipes, redirects, batch scripts. |

---

### Aider (`aider_run`)

| Tool | Rating | Notes |
|------|--------|-------|
| `aider_run` | ★★★★☆ | **AI coding assistant.** Pass `instruction` (what to change) + comma‑separated `files`. Default model: `deepseek/deepseek-v4-flash`. **Use case:** complex multi‑file refactoring, boilerplate generation, or when `fs_aedit`/`fs_edit_blocks` become tedious. **Workflow:** formulate precise instruction → list files → call `aider_run`. Always verify changes with `git_diff` afterward. |

---

### Vision (`vision_*`)

| Tool | Rating | Notes |
|------|--------|-------|
| `vision_analyze` | ★★★★☆ | **Analyze a local image file** using Qwen VL (`qwen3-vl-plus`). Takes `image_path` and optional `query`. Returns detailed description. Requires `QWEN_API_KEY` and `QWEN_API_ENDPOINT` env vars. |
| `vision_analyze_url` | ★★★★☆ | **Analyze an image from URL** — same model, same parameters. Fetches the image remotely. |

---

### Meta

| Tool | Rating | Notes |
|------|--------|-------|
| `reload_tools` | ★★★★★ | **Always call first after adding or modifying tools.** Hot‑reloads all built‑in modules and re‑runs `register_all()`. Reports "Reloaded N module(s)" and shows the tool list. |
| `ping` | ★★★★★ | Simple health‑check. Returns "pong" with a timestamp. |

---

## Tool Execution

- **All tools are asynchronous** — independent calls can be executed in parallel for efficiency.
- **Return value:** every tool returns a **string** result. On error, the tool returns a descriptive error message (exceptions are never propagated to the agent).
- **Hot‑reload:** after creating or editing any tool module, call `reload_tools` to load the changes without restarting the agent.

---

## General Best Practices

**Editing files:** always `fs_read` first, then use `fs_aedit` or `fs_edit_blocks` with `dry_run=True` to preview changes, then apply without `dry_run`.
**Searching:** for `fs_grep`, prefer targeting a specific file first; then use recursive search if needed. Remember the 1 MB size limit.
**Git:** use `git_diff` to review changes.
**New tools:** after adding any new tool module, immediately call `reload_tools` to make it available.
**Aider workflow:** identify the need → formulate precise instruction → list target files → call `aider_run`. Always verify changes with `git_diff` afterward. Do not use Aider for single‑file trivial edits — `fs_aedit`/`fs_edit_blocks` are faster.
**Audio / Transcription:** use `yandex_transcribe` for transcribing voice messages (OGG/MP3/WAV). Use `url` mode for CDN links (no disk write) or `audio_path` for local files.
**Vision:** use `vision_analyze` for local images, `vision_analyze_url` for URL-based images, `analyze_dynamic_scene` to capture the screen in motion. Supported formats: JPEG, PNG, GIF, WEBP. Powered by Qwen VL (`qwen3-vl-plus`).
    Ask a specific question in the `query` parameter, based on the context of the interaction, about what you expect to see in the given visual content. 
**Scheduler:** use `schedule_task` for periodic/autonomous tasks. Tasks fire on the next user message after `delay_minutes`. Cancel with `cancel_scheduled_task`.
**MAX Messenger specifics:**
    - **Always obtain `chat_id`** from the user's message context (shown as `Current chat_id: ...`). Do not guess or hardcode chat IDs unless explicitly provided.
    - **Prefer `max_send_message` for textual replies** — use Markdown formatting (`format="markdown"`) for readability, unless you need to send raw code or logs, then use `format="plain"`.
    - **For file sharing:** use `max_send_file` with an absolute path. The file must exist on the filesystem; the tool handles upload and attachment automatically.
    - **For voice replies:** use `max_send_voice`. Ensure the `text` is short, plain, and free of punctuation that might break TTS; prefer natural language.
    - **When a user sends a file**, the agent will see a `file_token` in the update. `max_download_file` may return 404 — use a Python script with the CDN URL as workaround.
    - **Remember that `chat_id` is a numeric string** (e.g., `"123456789"`) — pass it as a string in the tool call, and the tool will convert it to integer internally.
    - **Stickers:** to send a sticker, use `rest_api_call` with `method=POST`, endpoint `/messages?chat_id=X`, body `{"attachments": [{"type": "sticker", "payload": {"code": "..."}}]}` (no `text` field — sending sticker with text returns 400).
**Shell execution specifics:**
    To avoid common pitfalls, follow these guidelines when constructing the `command` argument:
        - **Use raw strings for complex commands** – avoid escaping backslashes; pass the command exactly as you would type it in `cmd.exe`.
        - **Keep commands short** – stay well below 8191 characters. For long data, use temporary files.
        - **Avoid interactive commands** – do not use `pause`, `set /p`, or commands that require keyboard input, as they will hang until the timeout.
        - **Handle quotes and spaces** – wrap file paths or arguments containing spaces in double quotes, e.g., `"C:\\Program Files\\MyApp\\app.exe"`.
        - **Use redirection and piping normally** – `dir /b > list.txt`, `type file.txt | find "string"`, etc., are fully supported.
        - **Set environment variables if needed** – you can prepend `set VAR=value &` to the command.
        - **Expect UTF‑8 output** – the function forces code page 65001, so non‑ASCII text (Cyrillic, Chinese, etc.) will be correctly returned.
        - **Be aware of timeouts** – long‑running commands will be terminated. Increase `timeout` if necessary (e.g., for `ping` or network operations).
        - **Handle return codes** – check the `returncode` in the JSON response; `0` usually means success.
---

This guide serves as the authoritative reference for tool capabilities, reliability, and safe usage.

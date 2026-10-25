"""Additional tools for MASTERMIND v2."""

import os
import sys
import json
import asyncio
from typing import Optional
from loguru import logger


def _smart_decode(data: bytes) -> str:
    """Decode bytes trying UTF-8 first, fallback to cp1251 (Russian Windows) on failure."""
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    try:
        return data.decode('cp1251')
    except UnicodeDecodeError:
        pass
    return data.decode('utf-8', errors='replace')


import asyncio
import json
import os
import sys
import subprocess

# Maximum command-line length for CreateProcess on Windows
_MAX_CMD_LEN = 32767

async def exec_python(parameter: str, timeout: int = 90) -> str:
    """⚠️ WARNING: Full system access. Use ONLY on explicit user request.

    Execute arbitrary Python code.
    Supports:
      - -c "code"   → inline code (quotes are automatically stripped)
      - file.py     → run a Python file
      - raw code    → auto-wrapped as -c

    Returns JSON: {"stdout": "...", "stderr": "...", "returncode": int}
    """
    # 1. Determine execution mode and sanitize the argument
    args = None
    if parameter.startswith('-c '):
        # Strip the -c prefix, then remove leading/trailing whitespace
        code = parameter[3:].strip()
        # Remove outermost matching quotes (single or double)
        if (code.startswith('"') and code.endswith('"')) or \
           (code.startswith("'") and code.endswith("'")):
            code = code[1:-1]
        args = ['-c', code]
    elif os.path.isfile(parameter):
        args = [parameter]
    elif os.path.isfile(os.path.join(os.getcwd(), parameter)):
        args = [os.path.join(os.getcwd(), parameter)]
    else:
        # Treat as raw code (no quotes to strip)
        args = ['-c', parameter]

    # 2. Prevent overly long command lines
    total_cmd = sys.executable + ' ' + ' '.join(args)
    if len(total_cmd) > _MAX_CMD_LEN:
        return json.dumps({
            "stdout": "",
            "stderr": f"Command exceeds Windows limit of {_MAX_CMD_LEN} characters",
            "returncode": -1,
        }, ensure_ascii=False)

    # 3. Force UTF-8 output for Python
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"  # Python 3.7+ global UTF-8 mode

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )

        # With PYTHONIOENCODING=utf-8, output is guaranteed UTF-8.
        # We use errors='replace' as a safety net.
        return json.dumps({
            "stdout": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace'),
            "returncode": proc.returncode if proc.returncode is not None else -1,
        }, ensure_ascii=False)

    except asyncio.TimeoutError:
        if proc:
            # Gracefully terminate, then kill if necessary
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        return json.dumps({
            "stdout": "",
            "stderr": f"Timeout ({timeout}s) exceeded",
            "returncode": -1,
        }, ensure_ascii=False)

    except Exception as e:
        if proc:
            proc.kill()
            await proc.wait()
        return json.dumps({
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1,
        }, ensure_ascii=False)


async def exec_shell(command: str, timeout: int = 90) -> str:
    """Execute an arbitrary shell command on the Linux host (via /bin/sh).

    - UTF-8 output; returns JSON: stdout, stderr, returncode.
    - Timeout kills the process.
    - ⚠️ WARNING: Full system access. Use ONLY on explicit user request.

    Args:
        command: Shell command string (e.g., "ls -la", "df -h", "echo hello")
        timeout: Execution timeout in seconds (default: 90)

    Returns:
        JSON string with stdout, stderr, returncode
    """
    # 1. Prevent overly long commands
    if len(command) > 8190:
        return json.dumps({
            "stdout": "",
            "stderr": "Command exceeds 8191 characters",
            "returncode": -1,
        }, ensure_ascii=False)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return json.dumps({
            "stdout": _decode_output(stdout),
            "stderr": _decode_output(stderr),
            "returncode": proc.returncode if proc.returncode is not None else -1,
        }, ensure_ascii=False)

    except asyncio.TimeoutError:
        if proc:
            proc.kill()
            await proc.wait()  # allow process to exit
        return json.dumps({
            "stdout": "",
            "stderr": f"Timeout ({timeout}s) exceeded",
            "returncode": -1,
        }, ensure_ascii=False)

    except Exception as e:
        if proc:
            proc.kill()
            await proc.wait()
        return json.dumps({
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1,
        }, ensure_ascii=False)


def _decode_output(data: bytes) -> str:
    """Decode bytes with fallback encodings."""
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # OEM code page (e.g., CP437)
            return data.decode(sys.stdout.encoding or 'cp437')
        except UnicodeDecodeError:
            # Never fails
            return data.decode('latin-1', errors='replace')


async def schedule_task(prompt: str, delay_minutes: int = 10, chat_id: str = "") -> str:
    """Schedule an LLM self-call. The scheduler fires the prompt automatically
    after delay_minutes (15 s tick) with no incoming message; the result goes to
    chat_id, or to the reasoning chat when chat_id is empty."""
    from lib.scheduler import get_scheduler
    sched = get_scheduler()
    task = sched.add_task(prompt, delay_minutes, chat_id or None)
    return json.dumps({
        "ok": True,
        "task_id": task["id"],
        "prompt": task["prompt"][:100],
        "run_at": task["run_at_str"],
        "message": f"Task {task['id']} scheduled at {task['run_at_str']}"
    }, ensure_ascii=False)


async def list_scheduled_tasks() -> str:
    """List all pending scheduled tasks, optionally filtered by chat_id."""
    from lib.scheduler import get_scheduler
    sched = get_scheduler()
    tasks = sched.list_tasks()
    if not tasks:
        return json.dumps({"ok": True, "tasks": [], "message": "No pending tasks"}, ensure_ascii=False)
    result = [{"id": t["id"], "prompt": t["prompt"][:120], "run_at": t["run_at_str"], "chat_id": t.get("chat_id")} for t in tasks]
    return json.dumps({"ok": True, "tasks_count": len(result), "tasks": result}, ensure_ascii=False)


async def cancel_scheduled_task(task_id: str) -> str:
    """Cancel a pending scheduled task by its ID."""
    from lib.scheduler import get_scheduler
    sched = get_scheduler()
    ok = sched.cancel_task(task_id)
    if ok:
        return json.dumps({"ok": True, "task_id": task_id, "message": "Task cancelled"}, ensure_ascii=False)
    return json.dumps({"ok": False, "task_id": task_id, "message": "Not found or already completed"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Fire-and-forget tool tasks (receipt + check_tool)
# ---------------------------------------------------------------------------

import time
import uuid as _uuid
from typing import Any as _Any, Dict as _Dict

RESULT_CHAR_LIMIT = 2000      # keep tool output (and context) bounded


def _truncate_result(text: str, limit: int = RESULT_CHAR_LIMIT) -> str:
    """Cut a tool result to ``limit`` chars with an explicit tail marker."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n[... truncated: shown first {limit} of {len(text)} chars; "
          f"narrow the query or ask for the rest if needed]"
    )


_TOOL_TASKS: _Dict[str, dict] = {}
_RUNNING_TASKS: set[asyncio.Task] = set()
_TASKS_TTL = 3600.0          # forget finished tasks after an hour
_MAX_FINISHED = 200          # hard cap on remembered finished tasks


def _track_task(task: asyncio.Task) -> None:
    """Keep a strong reference so the event loop cannot GC the task."""
    _RUNNING_TASKS.add(task)
    task.add_done_callback(_RUNNING_TASKS.discard)


def _prune_tasks() -> None:
    """Drop old finished tasks so the dict cannot grow forever."""
    now = time.time()
    finished = [(tid, t) for tid, t in _TOOL_TASKS.items() if t["status"] != "running"]
    excess = len(finished) - _MAX_FINISHED
    for tid, _ in finished[:max(0, excess)]:
        _TOOL_TASKS.pop(tid, None)
    for tid, t in list(_TOOL_TASKS.items()):
        if t["status"] != "running" and now - t.get("finished_at", now) > _TASKS_TTL:
            _TOOL_TASKS.pop(tid, None)


def _current_voice_session_id() -> str | None:
    try:
        from server.context import get_request_context
    except Exception:
        return None
    ctx = get_request_context()
    return ctx.voice_session_id if ctx is not None else None


def _task_matches_current_session(task: dict) -> bool:
    session_id = _current_voice_session_id()
    return bool(session_id and task.get("voice_session_id") == session_id)


def _describe_task(tid: str, t: dict, with_result: bool = False) -> str:
    age = round(time.time() - t["started_at"], 1)
    if t.get("delivered"):
        delivery = "delivered"
    elif t.get("consumed"):
        delivery = "claimed"
    else:
        delivery = "pending"
    head = (
        f"task_id={tid} | tool={t['tool']} | status={t['status']} | "
        f"elapsed={age}s | delivery={delivery}"
    )
    if with_result:
        if t["status"] == "ready":
            return head + f"\nRESULT:\n{t['result']}"
        if t["status"] == "failed":
            return head + f"\nERROR: {t['error']}"
    return head


async def _deliver_tool_result(task: dict, tid: str) -> None:
    """Push a finished background result into the agent queue as its own turn.

    Delivery is exactly-once: a result claimed by ``check_tool`` is not
    enqueued, and an automatically delivered result cannot be claimed again.
    For a voice session, speech is scheduled asynchronously by ``VoiceReply``;
    this coroutine must not wait for playback.
    """
    if task.get("consumed") or task.get("delivered"):
        return

    try:
        from server.runtime import runtime as rt
    except Exception as e:
        logger.warning(f"cannot enqueue tool result {tid}: {e}")
        return
    if rt is None or getattr(rt, "queue", None) is None:
        return

    from server.queue import Request
    from server.routing import NullReply, MaxReply, VoiceReply

    chat_id = task.get("chat_id")
    client_id = task.get("client_id")
    voice_session_id = task.get("voice_session_id")
    reply = NullReply()
    if chat_id and getattr(rt, "bot", None) is not None:
        try:
            reply = MaxReply(rt.bot, chat_id)
        except Exception:
            reply = NullReply()
    elif client_id:
        try:
            from server.voice import voice_hub
            if voice_session_id and not voice_hub.is_live(
                str(client_id), session_id=voice_session_id
            ):
                logger.info(
                    f"tool task {tid}: voice session {voice_session_id} "
                    "is gone; result not spoken"
                )
                task["delivered"] = True
                return
            reply = VoiceReply(
                voice_hub,
                str(client_id),
                session_id=voice_session_id,
            )
        except Exception as e:
            logger.info(f"voice reply unavailable for {client_id}: {e}")
            reply = NullReply()

    payload = task.get("result") if task["status"] == "ready" else task.get("error")
    req = Request(
        text=f"[Фоновый результат инструмента {task['tool']}]",
        source="tool_result",
        chat_id=chat_id,
        client_id=client_id,
        voice_session_id=voice_session_id,
        trusted=True,
        tool_result=str(payload),
        tool_task_id=tid,
        tool_name=task["tool"],
        reply=reply,
    )
    task["delivered"] = True
    try:
        await rt.queue.put(req)
    except Exception:
        task["delivered"] = False
        raise
    logger.info(
        f"tool task {tid} ({task['tool']}) -> queue as request {req.id} "
        f"(status={task['status']}, queue left={rt.queue.qsize()})"
    )


async def _run_tool_task(tid: str, tool: str, kwargs: dict) -> None:
    """Background runner: executes the tool, stores and delivers its outcome."""
    from tool_registry import get_registry
    task = _TOOL_TASKS.get(tid)
    if task is None:
        return
    try:
        registry = get_registry()
        result = await registry.call_tool(tool, **kwargs)
        raw = _truncate_result(str(result))
        task["status"] = "failed" if raw.startswith("Error") else "ready"
        task["result"] = raw
        if raw.startswith("Error"):
            task["error"] = raw
    except Exception as e:
        task["status"] = "failed"
        task["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"tool task {tid} ({tool}) failed: {e}")
    finally:
        task["finished_at"] = time.time()
        _prune_tasks()
        if task.get("auto_deliver", True):
            try:
                await _deliver_tool_result(task, tid)
            except Exception:
                logger.exception(f"tool task {tid}: deferred delivery failed")


async def start_tool(tool: str, arguments: str = "{}") -> str:
    """Launch any tool in the background and return a receipt immediately.

    The agent does not wait for the tool: it gets a task_id back and keeps
    talking. The result is delivered automatically as a separate turn. Use
    ``check_tool`` only for a status peek or to claim the result early.
    """
    from tool_registry import get_registry
    from server.context import get_request_context, is_voice_context

    if not is_voice_context():
        return ("Error: start_tool is available only in a voice turn. "
                "Call the tool directly instead — waiting for its result is fine here.")
    ctx = get_request_context()
    registry = get_registry()
    if registry.get_tool(tool) is None:
        return f"Error: tool '{tool}' not found"

    try:
        kwargs = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else {}
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON in arguments — {e}"
    if not isinstance(kwargs, dict):
        return "Error: arguments must be a JSON object"

    tid = _uuid.uuid4().hex[:8]
    _TOOL_TASKS[tid] = {
        "tool": tool,
        "args": kwargs,
        "status": "running",
        "result": None,
        "error": None,
        "started_at": time.time(),
        # where the finished result should be delivered
        "chat_id": getattr(ctx, "chat_id", None),
        "client_id": getattr(ctx, "client_id", None),
        "voice_session_id": getattr(ctx, "voice_session_id", None),
        "source": getattr(ctx, "source", None),
        "auto_deliver": True,
        "delivered": False,
        "consumed": False,
    }
    _track_task(asyncio.create_task(_run_tool_task(tid, tool, kwargs)))
    logger.info(
        f"start_tool: {tool}({json.dumps(kwargs, ensure_ascii=False)[:120]}) "
        f"-> task {tid}"
    )
    return (
        f"RECEIPT: task_id={tid} | tool={tool} | status=running\n"
        "Результат придёт автоматически отдельной репликой, когда будет готов. "
        "Не повторяй его. check_tool — только для мгновенного статуса или "
        "досрочного получения."
    )


async def check_tool(task_id: str) -> str:
    """Instant status check; claims an undelivered ready result.

    There is deliberately no waiting: a voice turn must stay responsive.
    If the result was already delivered or claimed earlier, do not repeat it.
    """
    tid = (task_id or "").strip()
    task = _TOOL_TASKS.get(tid)
    if task is None or not _task_matches_current_session(task):
        return f"Error: task '{tid}' not found (expired or belongs to another session)"

    if task["status"] == "running":
        return _describe_task(tid, task, with_result=False)
    if task.get("delivered"):
        return (
            _describe_task(tid, task, with_result=False)
            + "\nResult was already delivered automatically; do not repeat it."
        )
    if task.get("consumed"):
        return (
            _describe_task(tid, task, with_result=False)
            + "\nResult was already claimed; do not repeat it."
        )

    task["consumed"] = True
    task["auto_deliver"] = False
    return _describe_task(tid, task, with_result=True) + "\n(claimed)"


async def list_tool_tasks() -> str:
    """List background tool tasks belonging to the current voice session."""
    session_id = _current_voice_session_id()
    if not session_id:
        return "No background tool tasks."
    tasks = [
        (tid, task)
        for tid, task in _TOOL_TASKS.items()
        if task.get("voice_session_id") == session_id
    ]
    if not tasks:
        return "No background tool tasks."
    lines = [
        _describe_task(tid, task)
        for tid, task in sorted(tasks, key=lambda kv: kv[1]["started_at"])
    ]
    running = sum(1 for _, task in tasks if task["status"] == "running")
    return f"tasks={len(lines)} | running={running}\n" + "\n".join(lines)


TOOL_DEFINITIONS = [
    ("exec_python", exec_python, "⚠️ WARNING: Full system access. Use ONLY on explicit user request. Execute arbitrary Python code for quick testing and debugging. "
     "Pass code as: -c \"print('hello')\", or a .py filename, or raw code string.", {
        "type": "object",
        "properties": {
            "parameter": {
                "type": "string",
                "description": "Code to execute: -c \"code\" for inline, .py filename, or raw code"
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 30)",
                "default": 30
            }
        },
        "required": ["parameter"],
    }),
    ("exec_shell", exec_shell, "⚠️ WARNING: Full system access. Use ONLY on explicit user request. Execute arbitrary shell command on Linux (sh -c). "
     "⚠️ Full shell access. Supports pipes, redirects, shell commands.", {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command (e.g., 'dir /b', 'ipconfig', 'type file.txt')"
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 30)",
                "default": 30
            }
        },
        "required": ["command"],
    }),
    ("schedule_task", schedule_task, "Schedule an LLM self-call with arbitrary prompt after delay_minutes. The scheduler fires it automatically (15 s tick) with no incoming message; the result goes to chat_id, or to the reasoning chat when empty.", {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The prompt/question to ask the LLM when the time comes"},
            "delay_minutes": {"type": "integer", "description": "Delay in minutes before execution (default: 10)"},
            "chat_id": {"type": "string", "description": "Optional MAX chat_id the result should be reported to"},
        },
        "required": ["prompt"],
    }),
    ("list_scheduled_tasks", list_scheduled_tasks, "List all pending scheduled tasks.", {
        "type": "object",
        "properties": {},
    }),
    ("cancel_scheduled_task", cancel_scheduled_task, "Cancel a pending scheduled task by its ID.", {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "ID of the task to cancel"},
        },
        "required": ["task_id"],
    }),
    ("start_tool", start_tool, "Launch ANY tool in the background (fire-and-forget) and get a task_id immediately, without waiting for the result. Use it for every slow/network/vision/MAX/TTS/exec tool during a voice turn. The result is delivered automatically as a separate spoken turn; do not repeat it.", {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "description": "Name of the tool to run in background (e.g. tavily_search, exec_shell, client_record_audio)"},
            "arguments": {"type": "string", "description": "JSON object string with tool arguments, e.g. {\"query\": \"...\"} (default: {})"},
        },
        "required": ["tool"],
    }),
    ("check_tool", check_tool, "Instant status check for a background task started via start_tool. No waiting. A ready, not-yet-delivered result is claimed and returned once; an already delivered/claimed result is reported without repeating it.", {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id returned by start_tool"},
        },
        "required": ["task_id"],
    }),
    ("list_tool_tasks", list_tool_tasks, "List background tool tasks belonging to the current voice session, with status and delivery state.", {
        "type": "object",
        "properties": {},
    }),
]


def register_all(registry):
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} additional tools")

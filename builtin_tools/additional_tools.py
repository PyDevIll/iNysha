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

async def exec_python(parameter: str, timeout: int = 30) -> str:
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


async def exec_shell(command: str, timeout: int = 30) -> str:
    """Execute arbitrary shell command on Windows 10 using cmd.exe /c.

    - Forces UTF-8 output via `chcp 65001`.
    - Returns JSON: stdout, stderr, returncode.
    - Timeout kills the process.
    - ⚠️ WARNING: Full system access. Use ONLY on explicit user request.

    Args:
        command: Shell command string (e.g., "dir /b", "ipconfig", "echo hello")
        timeout: Execution timeout in seconds (default: 30)

    Returns:
        JSON string with stdout, stderr, returncode
    """
    # 1. Prevent overly long commands
    if len(command) > 8190:
        return json.dumps({
            "stdout": "",
            "stderr": "Command exceeds cmd.exe limit of 8191 characters",
            "returncode": -1,
        }, ensure_ascii=False)

    # 2. Force UTF-8 output
    cmd = f"chcp 65001 >nul & {command}"

    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
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


async def schedule_task(prompt: str, delay_minutes: int = 10) -> str:
    """Schedule an LLM self-call. On the next user message after delay expires,
    the prompt is injected as if the user asked it. No external scheduler needed."""
    from scheduler import get_scheduler
    sched = get_scheduler()
    task = sched.add_task(prompt, delay_minutes, chat_id)
    return json.dumps({
        "ok": True,
        "task_id": task["id"],
        "prompt": task["prompt"][:100],
        "run_at": task["run_at_str"],
        "message": f"Task {task['id']} scheduled at {task['run_at_str']}"
    }, ensure_ascii=False)


async def list_scheduled_tasks() -> str:
    """List all pending scheduled tasks, optionally filtered by chat_id."""
    from scheduler import get_scheduler
    sched = get_scheduler()
    tasks = sched.list_tasks()
    if not tasks:
        return json.dumps({"ok": True, "tasks": [], "message": "No pending tasks"}, ensure_ascii=False)
    result = [{"id": t["id"], "prompt": t["prompt"][:120], "run_at": t["run_at_str"], "chat_id": t["chat_id"]} for t in tasks]
    return json.dumps({"ok": True, "tasks_count": len(result), "tasks": result}, ensure_ascii=False)


async def cancel_scheduled_task(task_id: str) -> str:
    """Cancel a pending scheduled task by its ID."""
    from scheduler import get_scheduler
    sched = get_scheduler()
    ok = sched.cancel_task(task_id)
    if ok:
        return json.dumps({"ok": True, "task_id": task_id, "message": "Task cancelled"}, ensure_ascii=False)
    return json.dumps({"ok": False, "task_id": task_id, "message": "Not found or already completed"}, ensure_ascii=False)


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
    ("exec_shell", exec_shell, "⚠️ WARNING: Full system access. Use ONLY on explicit user request. Execute arbitrary shell command on Windows 10 (cmd.exe /c). "
     "⚠️ Full shell access. Supports pipes, redirects, batch commands.", {
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
    ("schedule_task", schedule_task, "Schedule an LLM self-call with arbitrary prompt after delay_minutes. On the next user message after the delay, the prompt is injected as if the user asked it.", {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The prompt/question to ask the LLM when the time comes"},
            "delay_minutes": {"type": "integer", "description": "Delay in minutes before execution (default: 10)"},
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
]



def register_all(registry):
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} additional tools")

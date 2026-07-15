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


async def ping() -> str:
    """Simple ping/pong health check. Returns 'pong' with current timestamp."""
    from datetime import datetime
    return json.dumps({
        "result": "pong",
        "timestamp": datetime.now().isoformat(),
    }, ensure_ascii=False)


async def exec_python(parameter: str, timeout: int = 30) -> str:
    """WARNING: Full system execution access. This tool should ONLY be used when explicitly requested by the user for debugging or system tasks.

    Execute arbitrary Python code for quick testing and debugging.

    Parameter can be:
      - `-c "print('hello')"` — inline code (like python -c)
      - A `.py` filename (absolute or relative to agent's CWD)
      - Raw Python code (auto-wrapped as -c)
    
    Returns JSON: {"stdout": "...", "stderr": "..."}
    """
    # Determine execution mode
    if parameter.startswith('-c '):
        code = parameter[3:]
        args = ['-c', code]
    elif os.path.isfile(parameter):
        args = [parameter]
    elif os.path.isfile(os.path.join(os.getcwd(), parameter)):
        args = [os.path.join(os.getcwd(), parameter)]
    else:
        # Treat as raw code
        args = ['-c', parameter]

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return json.dumps({
            "stdout": _smart_decode(stdout),
            "stderr": _smart_decode(stderr),
            "returncode": proc.returncode,
        }, ensure_ascii=False)
    except asyncio.TimeoutError:
        proc.kill()
        return json.dumps({
            "stdout": "",
            "stderr": f"Timeout ({timeout}s) exceeded",
            "returncode": -1,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1,
        }, ensure_ascii=False)


async def exec_shell(command: str, timeout: int = 30) -> str:
    """WARNING: Full system execution access. This tool should ONLY be used when explicitly requested by the user for debugging or system tasks.

    Execute arbitrary shell command on Windows 10 using cmd.exe /c.

    ⚠️ DANGEROUS: Full shell access. Use with extreme caution.
    - Runs via cmd.exe /c {command}
    - Supports batch commands, pipes, redirects (>, |, &&)
    - Returns JSON: stdout, stderr, returncode
    - Timeout kills the process

    Args:
        command: Shell command string (e.g., "dir /b", "ipconfig", "echo hello")
        timeout: Execution timeout in seconds (default: 30)

    Returns:
        JSON string with stdout, stderr, returncode
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return json.dumps({
            "stdout": _smart_decode(stdout),
            "stderr": _smart_decode(stderr),
            "returncode": proc.returncode,
        }, ensure_ascii=False)
    except asyncio.TimeoutError:
        proc.kill()
        return json.dumps({
            "stdout": "",
            "stderr": f"Timeout ({timeout}s) exceeded",
            "returncode": -1,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1,
        }, ensure_ascii=False)


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
    ("ping", ping, "Simple ping/pong health check. Returns pong with current timestamp.", {
        "type": "object",
        "properties": {},
        "required": [],
    }),
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

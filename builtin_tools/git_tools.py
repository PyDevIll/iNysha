"""Git operations tools for MASTERMIND v2 — async subprocess-based."""

import asyncio
import os
from pathlib import Path
from typing import Optional
import shlex

from loguru import logger


async def _run_git(path: str, *args: str, timeout: int = 30) -> str:
    """Run a git command asynchronously and return stdout+stderr."""
    p = Path(path).resolve()
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(p),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        result = stdout.decode('utf-8', errors='replace')
        if proc.returncode != 0:
            err = stderr.decode('utf-8', errors='replace')
            result += f"\n[stderr]: {err}"
        return result.strip() or "(no output)"
    except asyncio.TimeoutError:
        return f"Git command timed out ({timeout}s)"


async def git_status(path: str = ".") -> str:
    """Show working tree status."""
    return await _run_git(path, "status", "--short")


async def git_log(path: str = ".", count: int = 10, extra_args: str = "") -> str:
    """Show last N commit logs (oneline format)."""
    extra = shlex.split(extra_args)
    return await _run_git(path, "log", *extra, f"-{count}", "--oneline")


async def git_diff(path: str = ".", staged: bool = False) -> str:
    """Show changes (unified diff format). Use staged=True for --cached."""
    args = ["diff", "--cached" if staged else None]
    args = [a for a in args if a]  # filter None
    return await _run_git(path, *args)


TOOL_DEFINITIONS = [
    ("git_status", git_status, "Show working tree status", {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repository path (default: .)"},
        },
    }),
    ("git_log", git_log, "Show last N commit logs (oneline format)", {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repository path (default: .)"},
            "count": {"type": "integer", "description": "Number of commits (default: 10)"},
            "extra_args": {"type": "string", "description": "Extra git log arguments, e.g. '--reverse --format=\"%ai %s\"'"},
        },
    }),
    ("git_diff", git_diff, "Show changes (unified diff format). Use staged=True for --cached", {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repository path (default: .)"},
            "staged": {"type": "boolean", "description": "Show staged changes (--cached) (default: false)"},
        },
    })
]


def register_all(registry):
    """Register all git tools with the given registry."""
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} git tools")

"""In-process task scheduler for MASTERMIND v2.
Persists tasks to data/scheduler_tasks.json.
"""

import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

DATA_DIR = Path(__file__).resolve().parent / "data"
TASKS_FILE = DATA_DIR / "scheduler_tasks.json"

# Unique ID counter (thread-safe)
_id_lock = threading.Lock()
_next_id = 0


def _generate_id() -> str:
    global _next_id
    with _id_lock:
        _next_id += 1
        return f"task_{int(time.time())}_{_next_id}"


class Scheduler:
    """Singleton scheduler. Manages persistent delayed tasks."""

    _instance: Optional["Scheduler"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._tasks: list[dict] = []
        self._load()
        logger.info(f"Scheduler initialized: {len(self._tasks)} pending task(s)")

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if TASKS_FILE.exists():
            try:
                data = TASKS_FILE.read_text(encoding="utf-8")
                self._tasks = json.loads(data)
            except Exception as e:
                logger.error(f"Failed to load scheduler tasks: {e}")
                self._tasks = []
        else:
            self._tasks = []

    def _save(self) -> None:
        TASKS_FILE.write_text(
            json.dumps(self._tasks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── public API ───────────────────────────────────────────────

    def add_task(self, prompt: str, delay_minutes: int) -> dict:
        """Schedule a task. Returns the task dict."""
        task = {
            "id": _generate_id(),
            "prompt": prompt,
            "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "run_at": int(time.time()) + delay_minutes * 60,
            "run_at_str": datetime.fromtimestamp(
                int(time.time()) + delay_minutes * 60
            ).strftime("%d.%m.%Y %H:%M"),
            "status": "pending",
        }
        self._tasks.append(task)
        self._save()
        logger.info(f"Scheduled task {task['id']}: '{prompt[:60]}' in {delay_minutes}min")
        return task

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task by ID. Returns True if cancelled."""
        for task in self._tasks:
            if task["id"] == task_id and task["status"] == "pending":
                task["status"] = "cancelled"
                self._save()
                logger.info(f"Cancelled task {task_id}")
                return True
        return False

    def list_tasks(self) -> list[dict]:
        """List pending tasks, optionally filtered by chat_id."""
        result = []
        for task in self._tasks:
            if task["status"] == "pending":
                result.append(task)
        return result

    def get_due_tasks(self) -> list[dict]:
        """Get all pending tasks past their run_at time."""
        now = int(time.time())
        due = []
        for task in self._tasks:
            if task["status"] == "pending" and task["run_at"] <= now:
                due.append(task)
        return due

    def mark_done(self, task_id: str) -> bool:
        """Mark a task as completed. Returns True if found."""
        for task in self._tasks:
            if task["id"] == task_id and task["status"] == "pending":
                task["status"] = "done"
                self._save()
                logger.info(f"Task {task_id} marked done")
                return True
        return False


def get_scheduler() -> Scheduler:
    """Get the scheduler singleton."""
    return Scheduler()

"""Context Manager v4 — Hybrid Observation Masking + LLM Summarization + Layered Memory.

Based on Lindenbauer et al. (NeurIPS 2025): observation masking as first line,
LLM summarization as last resort. Structured in 4 memory layers with token budget.

Layers (ordered by proximity to LLM):
  L0_SCRATCHPAD  — current turn, full detail
  L1_SLIDING     — last N messages, preserved verbatim
  L2_MASKED      — older tool outputs replaced with [MASKED: ...]
  L3_COMPRESSED  — LLM-summarized batches (last resort)
  LX_PERSISTENT  — key facts surviving across sessions
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional

import tiktoken
from loguru import logger

# ── Tokenizer ──────────────────────────────────────────────────────────
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"tiktoken not available, falling back to char/3: {e}")
            _tokenizer = None
    return _tokenizer


def _token_count(text: str) -> int:
    """Return token count using tiktoken (fallback to char/3)."""
    tokenizer = _get_tokenizer()
    if tokenizer:
        tokens = tokenizer.encode_ordinary(text)
        return len(tokens)
    return max(1, len(text) // 3)


# ── Constants ──────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 100000          # soft cap; proactive trim at 80%
SLIDING_WINDOW_SIZE = 12            # last N messages kept verbatim
MASK_BATCH_SIZE = 12                 # mask tool outputs older than this many msgs
COMPRESSION_BATCH = 30              # trigger LLM summarization every N messages
PERSISTENT_FILE = "persistent_memory.json"
EMERGENCY_FILE = "emergency_save.json"
MAX_COMPRESSED_DICTS = 5            # merge compressed summaries when exceeding this

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Memory Layer enum ──────────────────────────────────────────────────
class Layer(IntEnum):
    SCRATCHPAD = 0
    SLIDING = 1
    MASKED = 2
    COMPRESSED = 3
    PERSISTENT = 4

# ── Data structures ────────────────────────────────────────────────────
@dataclass
class MemoryEntry:
    role: str
    content: str
    time: str = ""
    reasoning: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    tool_calls: list = field(default_factory=list)
    masked: bool = False
    layer: Layer = Layer.SLIDING


@dataclass
class TokenBudget:
    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def usage_ratio(self) -> float:
        return self.used / self.limit if self.limit > 0 else 0

    def count(self, text: str) -> int:
        return _token_count(text)


# ── Utility helper ─────────────────────────────────────────────────────
def _truncate_first_last(text: str, max_first: int = 200, max_last: int = 200) -> str:
    """Keep first max_first chars and last max_last chars, join with [...]."""
    if len(text) <= max_first + max_last + 20:
        return text
    first_part = text[:max_first]
    last_part = text[-max_last:]
    return f"{first_part}\n…[truncated {len(text) - max_first - max_last} chars]…\n{last_part}"


def _is_likely_json(text: str) -> bool:
    text = text.strip()
    if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    return False


# ── Observation Masker ─────────────────────────────────────────────────
class ObservationMasker:
    """Replaces verbose tool outputs with compact placeholders.

    Improved variant:
      - Keeps full content if length < 500 chars or valid JSON.
      - Truncates large outputs to first + last lines (200 chars each).
    """

    @staticmethod
    def mask(entry: MemoryEntry) -> MemoryEntry:
        if entry.role != "tool" or entry.masked:
            return entry

        content = entry.content or ""
        tool = entry.tool_name or "unknown"

        # Keep small outputs and JSON structures intact
        if len(content) < 500 or _is_likely_json(content):
            masked_content = content
        else:
            masked_content = _truncate_first_last(content, max_first=200, max_last=200)

        masked_entry = MemoryEntry(
            role=entry.role,
            content=masked_content,
            time=entry.time,
            tool_name=tool,
            tool_call_id=entry.tool_call_id,
            masked=True,
            layer=Layer.MASKED,
        )
        return masked_entry


# ── Persistent Store ───────────────────────────────────────────────────
class PersistentStore:
    """Key facts surviving across sessions (user identity, critical knowledge)."""

    def __init__(self, data_dir: Path):
        self._path = data_dir / PERSISTENT_FILE
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning(f"Persistent store load failed: {e}")
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding='utf-8')

    @property
    def facts(self) -> dict:
        return dict(self._data)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._save()

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(key, default)

    def as_context_string(self) -> str:
        if not self._data:
            return ""
        lines = ["## Persistent Memory"]
        for k, v in self._data.items():
            lines.append(f"- **{k}**: {v}")
        return "\n".join(lines) + "\n"


# ── Context Assembler ──────────────────────────────────────────────────
class ContextAssembler:
    """Assembles memory layers into a single context string for the LLM.

    Order (DeepSeek cache-optimized):
      system → persistent → compressed → masked → sliding → current
    """

    def __init__(self, token_limit: int = DEFAULT_MAX_TOKENS):
        self.budget = TokenBudget(limit=token_limit)

    def assemble(
        self,
        system_prompt: str,
        persistent: str,
        compressed: list[MemoryEntry],
        masked: list[MemoryEntry],
        sliding: list[MemoryEntry],
    ) -> tuple[str, TokenBudget]:
        """Build final context string. Returns (text, budget)."""
        self.budget = TokenBudget(limit=self.budget.limit)
        parts: list[str] = []

        # 1. System prompt (static → DeepSeek prefix cache)
        parts.append(system_prompt)
        self.budget.used += self.budget.count(system_prompt)

        # 2. Persistent facts
        if persistent:
            parts.append(persistent)
            self.budget.used += self.budget.count(persistent)

        # 3. Compressed history (LLM summaries)
        for entry in compressed:
            text = self._entry_to_text(entry)
            if self.budget.remaining > self.budget.count(text):
                parts.append(text)
                self.budget.used += self.budget.count(text)
                self.budget.used += self.budget.count(entry.reasoning)

        # 4. Masked observations
        for entry in masked:
            text = self._entry_to_text(entry)
            if self.budget.remaining > self.budget.count(text):
                parts.append(text)
                self.budget.used += self.budget.count(text)

        # 5. Sliding window (verbatim, newest first in prompt order)
        for entry in sliding:
            text = self._entry_to_text(entry)
            if self.budget.remaining > self.budget.count(text):
                parts.append(text)
                self.budget.used += self.budget.count(text)
                self.budget.used += self.budget.count(entry.reasoning)
            else:
                logger.warning("Token budget exhausted — truncating sliding window")

        return "\n\n".join(parts), self.budget

    @staticmethod
    def _entry_to_text(entry: MemoryEntry) -> str:
        """Convert a MemoryEntry to prompt text."""
        time_str = f"[{entry.time}]" if entry.time else ""
        if entry.role == "system":
            return entry.content
        elif entry.role == "assistant":
            text = f"**ASSISTANT** {time_str}"
            if entry.reasoning:
                text += f"\n*reasoning:* {entry.reasoning}"
            if entry.content:
                text += f"\n{entry.content}"
            return text
        elif entry.role == "tool":
            return f"**TOOL [{entry.tool_name}]** {time_str}\n{entry.content}"
        else:
            return f"**{entry.role.upper()}** {time_str}\n{entry.content}"


# ── ContextPool v3 ─────────────────────────────────────────────────────
class ContextPool:
    """Hybrid context manager with observation masking, LLM summarization, and crash recovery."""

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        window_size: int = SLIDING_WINDOW_SIZE,
        data_dir: Optional[Path] = None,
    ):
        self.max_tokens = max_tokens
        self.window_size = window_size
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Memory layers
        self._all_entries: list[MemoryEntry] = []   # master log
        self._compressed_dicts: list[dict] = []      # L3 structured summaries

        # Infrastructure
        self._masker = ObservationMasker()
        self._persistent = PersistentStore(self._data_dir)
        self._assembler = ContextAssembler(token_limit=max_tokens)
        self._save_counter: int = 0
        self.overflow: bool = False

        # Caches
        self._token_counts: dict[int, int] = {}   # id(entry) -> token count
        self._compress_lock = asyncio.Lock()
        self._save_lock = threading.Lock()

        # Attempt crash recovery
        self._restore_if_needed()

    # ── Properties ─────────────────────────────────────────────────
    @property
    def messages(self) -> list[dict]:
        """Return sliding window as OpenAI-format dicts for API compatibility."""
        return [self._entry_to_openai_dict(e) for e in self.sliding_window]

    @property
    def sliding_window(self) -> list[MemoryEntry]:
        """Last N entries in full detail."""
        return self._all_entries[-self.window_size:] if self._all_entries else []

    @property
    def masked_entries(self) -> list[MemoryEntry]:
        """Entries older than sliding window, with tool outputs masked."""
        if len(self._all_entries) <= self.window_size:
            return []
        older = self._all_entries[: -self.window_size]
        return [self._masker.mask(e) for e in older]

    @property
    def length(self) -> int:
        return len(self._all_entries)

    # ── Core API ───────────────────────────────────────────────────
    def get_context_length(self) -> int:
        """Estimate token count of entire context using tiktoken."""
        total = 0
        for entry in self._all_entries:
            total += self._token_count_entry(entry)
        return total

    def append(self, message: dict, save: bool = False) -> None:
        """Append a raw message dict."""
        entry = self._dict_to_entry(message)
        self._all_entries.append(entry)
        self._save_counter += 1
        self._token_counts.clear()          # invalidate cache

        if save:
            self._save_to_file(entry)

        self._check_overflow()
        self._auto_mask()
        self._periodic_save()

    def append_assistant_message(
        self, message: dict, save: bool = False, has_tool_calls: bool = False
    ) -> None:
        """Append assistant message with tool-call aware processing."""
        entry = MemoryEntry(
            role="assistant",
            content=message.get("content") if not has_tool_calls else None,
            reasoning=message.get("reasoning_content", ""),
            time=datetime.now().strftime("%d.%m.%Y, %H:%M"),
            tool_calls=message.get("tool_calls", []) if has_tool_calls else [],
        )
        self._all_entries.append(entry)
        self._save_counter += 1
        self._token_counts.clear()

        if save:
            self._save_to_file(entry)

        self._check_overflow()
        self._auto_mask()
        self._periodic_save()

    def assign_messages(self, messages: list[dict]) -> None:
        """Bulk load messages (e.g., from last memory)."""
        self._all_entries = [self._dict_to_entry(m) for m in messages]
        self._token_counts.clear()
        # No orphan deletion – tool messages are kept.

    def get_chat_history(self, messages: Optional[list[dict]] = None) -> str:
        """Render chat history as markdown string (for compression)."""
        entries = [self._dict_to_entry(m) for m in messages] if messages else self._all_entries
        lines = []
        for entry in entries:
            lines.append(f"\n**{entry.role}:** [{entry.time or '--:--'}]")
            if entry.reasoning:
                lines.append(f"*reasoning:* {entry.reasoning}")
            if entry.content:
                lines.append(entry.content)
            lines.append("---")
        return "\n".join(lines)

    def build_context(self, system_prompt: str = "") -> tuple[str, TokenBudget]:
        """Assemble full context from all layers. Returns (text, budget)."""
        persistent = self._persistent.as_context_string()
        # Convert compressed dicts to MemoryEntry list for assembly
        compressed_mems = [self._compressed_dict_to_entry(d) for d in self._compressed_dicts]
        logger.debug(f"build_context: compressed entries count = {len(self._compressed_dicts)}")
        return self._assembler.assemble(
            system_prompt=system_prompt,
            persistent=persistent,
            compressed=compressed_mems,
            masked=self.masked_entries,
            sliding=self.sliding_window,
        )

    # ── Compression ────────────────────────────────────────────────
    async def compress(self, helper_agent) -> Optional[str]:
        """LLM summarization — last resort for large batches (structured output + narrative)."""
        async with self._compress_lock:
            self.log_state("BEFORE compression")

            est_tokens = self.get_context_length()
            overflow_ratio = est_tokens / self.max_tokens if self.max_tokens > 0 else 0
            enough_entries = len(self._all_entries) >= 20
            token_overflow = overflow_ratio >= 0.70
            enough_old = (len(self._all_entries) - self.window_size) >= 10

            if not helper_agent:
                return None
            if not enough_entries and not token_overflow:
                return None
            if not enough_old and not token_overflow:
                return None
            if len(self._all_entries) < 3:
                return None

            logger.info(
                f"Context compression triggered: {len(self._all_entries)} entries, "
                f"{est_tokens} tokens ({overflow_ratio:.0%})"
            )

            # Take the oldest batch beyond sliding window
            batch = self._all_entries[: -self.window_size]
            if not batch:
                return None

            chat_text = self.get_chat_history(
                [self._entry_to_dict(e) for e in batch]
            )

            helper_message = {
                "role": "user",
                "name": "MASTERMIND",
                "content": (
                    "**ACTION**: COMPRESS\n\n"
                    "Output a **structured summary** with the following sections:\n"
                    "- **Key Facts:** (decisions, important data, user intent)\n"
                    "- **Tool Results:** (critical outputs, file contents, search snippets)\n"
                    "- **Decisions:** (what was decided/changed)\n"
                    "- **Narrative:** (a detailed, unstructured story of the interaction, describing the flow, user messages, agent reasoning, and actions)\n\n"
                    "Use bullet points for structured sections. Keep relevant details, discard noise.\n\n"
                    "**CONTENT**:\n" + chat_text
                ),
            }

            # Use helper agent for compression (async)
            try:
                helper_agent.messages.assign_messages([helper_message])
                response = await helper_agent.llm_request()
            except Exception as e:
                logger.error(f"Compression helper LLM call failed: {e}")
                return None

            # Extract compressed text
            if isinstance(response, dict):
                compressed_text = response["choices"][0]["message"]["content"]
            else:
                compressed_text = response.choices[0].message.content

            # Parse structured sections from LLM response (including narrative)
            parsed = self._parse_compressed_text(compressed_text)
            now = datetime.now().strftime("%d.%m.%Y, %H:%M")
            parsed["timestamp"] = now

            # Store as dict for structured access
            self._compressed_dicts.append(parsed)
            logger.info(
                f"Compressed entry added: structured dict length={len(parsed)} "
                f"content length={len(compressed_text)}"
            )

            # Trim old entries: keep only last window_size
            kept = self._all_entries[-self.window_size:]
            self._all_entries = kept
            self._token_counts.clear()
            # No orphan deletion – tool messages are kept.

            # Save to disk
            comp_file = self._data_dir / "compressed_history.txt"
            with open(comp_file, 'a', encoding='utf-8') as f:
                f.write(f"# Compressed {len(batch)} turns:\n{compressed_text}\n\n")

            last_file = self._data_dir / "last_compression.txt"
            with open(last_file, 'w', encoding='utf-8') as f:
                f.write(compressed_text + "\n\n")

            self.overflow = False
            self._check_overflow()
            logger.info(
                f"Compression complete: {len(batch)} turns → summary, "
                f"context now ~{self.get_context_length()} tokens"
            )

            # If we have too many compressed dicts, merge them into a meta-summary
            if len(self._compressed_dicts) >= MAX_COMPRESSED_DICTS:
                logger.info(f"Compressed dicts count {len(self._compressed_dicts)} exceeds threshold, merging...")
                await self.compress_compressed(helper_agent)

            self.log_state("AFTER compression")
            return compressed_text

    async def compress_compressed(self, helper_agent) -> Optional[str]:
        """
        Merge all existing compressed summaries into a single meta-summary.
        Replaces self._compressed_dicts with a list containing the new merged summary.

        ⚠️ This method must be called while `_compress_lock` is already acquired
        (i.e., from within `compress()`). It does not acquire the lock itself.
        """
        if len(self._compressed_dicts) <= 1:
            logger.debug("compress_compressed: only 1 summary, nothing to merge.")
            return None

        logger.info(f"compress_compressed: merging {len(self._compressed_dicts)} summaries...")

        # Build a text that concatenates all existing summaries (with separators)
        summary_texts = []
        for i, d in enumerate(self._compressed_dicts):
            parts = []
            if d.get("facts"):
                parts.append(f"Key Facts: {d['facts']}")
            if d.get("tool_results"):
                parts.append(f"Tool Results: {d['tool_results']}")
            if d.get("decisions"):
                parts.append(f"Decisions: {d['decisions']}")
            if d.get("narrative"):
                parts.append(f"Narrative: {d['narrative']}")
            if not parts and d.get("raw_text"):
                parts.append(d["raw_text"])
            summary_texts.append(f"--- Summary #{i + 1} ---\n" + "\n".join(parts))

        combined = "\n\n".join(summary_texts)

        helper_message = {
            "role": "user",
            "name": "MASTERMIND",
            "content": (
                    "**ACTION**: META‑COMPRESS\n\n"
                    "You are given multiple structured summaries of past interactions. "
                    "Synthesize them into a single comprehensive summary that captures the most important information.\n\n"
                    "Output a structured summary with these sections:\n"
                    "- **Key Facts:** (overarching decisions, important data, user intent)\n"
                    "- **Tool Results:** (critical tool outputs, key findings)\n"
                    "- **Decisions:** (major decisions made)\n"
                    "- **Narrative:** (a unified story of the entire interaction, describing the flow, user messages, agent reasoning, and actions)\n\n"
                    "Focus on high‑level insights, discard redundant details.\n\n"
                    "**CONTENT**:\n" + combined
            ),
        }

        try:
            helper_agent.messages.assign_messages([helper_message])
            response = await helper_agent.llm_request()
        except Exception as e:
            logger.error(f"Meta‑compression LLM call failed: {e}")
            return None

        if isinstance(response, dict):
            merged_text = response["choices"][0]["message"]["content"]
        else:
            merged_text = response.choices[0].message.content

        parsed = self._parse_compressed_text(merged_text)
        parsed["timestamp"] = datetime.now().strftime("%d.%m.%Y, %H:%M")

        # Replace the list with the single merged summary
        self._compressed_dicts = [parsed]
        self._emergency_save()
        logger.info(f"Meta‑compression complete: {len(self._compressed_dicts)} summary now.")
        return merged_text


    @staticmethod
    def _parse_compressed_text(text: str) -> dict:
        """Parse LLM response into structured dict with sections using robust regex."""
        # Define patterns for each section (case-insensitive, optional bold/asterisks)
        patterns = {
            "facts": r'\*\*Key Facts:\*\*\s*(.*?)(?=\*\*Tool Results:\*\*|\*\*Decisions:\*\*|\*\*Narrative:\*\*|$)',
            "tool_results": r'\*\*Tool Results:\*\*\s*(.*?)(?=\*\*Key Facts:\*\*|\*\*Decisions:\*\*|\*\*Narrative:\*\*|$)',
            "decisions": r'\*\*Decisions:\*\*\s*(.*?)(?=\*\*Key Facts:\*\*|\*\*Tool Results:\*\*|\*\*Narrative:\*\*|$)',
            "narrative": r'\*\*Narrative:\*\*\s*(.*?)(?=\*\*Key Facts:\*\*|\*\*Tool Results:\*\*|\*\*Decisions:\*\*|$)',
        }
        # Also try with leading dash (bullet) variants
        alt_patterns = {
            "facts": r'-\s*\*\*Key Facts:\*\*\s*(.*?)(?=-\s*\*\*Tool Results:\*\*|-\s*\*\*Decisions:\*\*|-\s*\*\*Narrative:\*\*|$)',
            "tool_results": r'-\s*\*\*Tool Results:\*\*\s*(.*?)(?=-\s*\*\*Key Facts:\*\*|-\s*\*\*Decisions:\*\*|-\s*\*\*Narrative:\*\*|$)',
            "decisions": r'-\s*\*\*Decisions:\*\*\s*(.*?)(?=-\s*\*\*Key Facts:\*\*|-\s*\*\*Tool Results:\*\*|-\s*\*\*Narrative:\*\*|$)',
            "narrative": r'-\s*\*\*Narrative:\*\*\s*(.*?)(?=-\s*\*\*Key Facts:\*\*|-\s*\*\*Tool Results:\*\*|-\s*\*\*Decisions:\*\*|$)',
        }

        sections = {"facts": "", "tool_results": "", "decisions": "", "narrative": ""}
        # Try primary patterns first
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
        # If any section missing, try alt patterns
        for key, pattern in alt_patterns.items():
            if not sections[key]:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    sections[key] = match.group(1).strip()

        # Fallback: if still empty, treat entire text as raw
        if not any(sections.values()):
            sections["raw_text"] = text
        else:
            sections["raw_text"] = text  # keep raw text for reference

        return sections

    # ── Crash Recovery ─────────────────────────────────────────────
    def _restore_if_needed(self) -> bool:
        """Called on init. Restores from emergency save if it exists."""
        save_file = self._data_dir / EMERGENCY_FILE
        if not save_file.exists():
            return False
        try:
            raw = json.loads(save_file.read_text(encoding='utf-8'))
            if "all_entries" in raw:
                self._all_entries = [self._dict_to_entry(m) for m in raw["all_entries"]]
            else:
                # backward-compat: whole file was list of entries
                self._all_entries = [self._dict_to_entry(m) for m in raw]
            if "compressed_dicts" in raw:
                self._compressed_dicts = raw["compressed_dicts"]
            else:
                self._compressed_dicts = []
            self._token_counts.clear()
            self._check_overflow()
            if self.overflow:
                logger.warning(
                    f"🔄 Restored {len(self._all_entries)} entries — overflow "
                    f"({self.get_context_length()} tokens), compression advised"
                )
            else:
                logger.info(f"🔄 Restored {len(self._all_entries)} messages from emergency save")
            return True
        except Exception as e:
            logger.error(f"Emergency restore failed: {e}")
            return False

    def _emergency_save(self) -> None:
        """Save full state as JSON (both entries and compressed dicts)."""
        with self._save_lock:
            save_file = self._data_dir / EMERGENCY_FILE
            try:
                raw = {
                    "all_entries": [self._entry_to_dict(e) for e in self._all_entries],
                    "compressed_dicts": self._compressed_dicts,
                }
                save_file.write_text(json.dumps(raw, ensure_ascii=False, default=str), encoding='utf-8')
                logger.debug(f"Emergency save: {len(raw['all_entries'])} entries, {len(raw['compressed_dicts'])} compressed")
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")

    def restore_from_emergency(self) -> bool:
        """Public API: manual restore."""
        return self._restore_if_needed()

    # ── Persistent facts ───────────────────────────────────────────
    def remember(self, key: str, value: str) -> None:
        """Store a persistent fact."""
        self._persistent.set(key, value)

    def recall(self, key: str) -> str:
        """Retrieve a persistent fact."""
        return self._persistent.get(key)

    # ── Internals ──────────────────────────────────────────────────
    def _dict_to_entry(self, msg: dict) -> MemoryEntry:
        """Convert raw dict to MemoryEntry. Preserves tool_calls for round-trip."""
        msg_time = msg.get("time", datetime.now().strftime("%d.%m.%Y, %H:%M"))
        if msg.get("content", None):
            msg_content = f"[Time: {msg_time}]" + msg["content"]
        else:
            msg_content = ""
        return MemoryEntry(
            role=msg.get("role", "unknown"),
            content=msg_content,
            reasoning=msg.get("reasoning_content", ""),
            time=msg_time,
            tool_name=msg.get("name", ""),
            tool_call_id=msg.get("tool_call_id", ""),
            tool_calls=msg.get("tool_calls", []),
        )

    def _compressed_dict_to_entry(self, d: dict) -> MemoryEntry:
        """Convert compressed dict to MemoryEntry for context assembly."""
        content_lines = []
        if d.get("facts"):
            content_lines.append(f"**Key Facts:**\n{d['facts']}")
        if d.get("tool_results"):
            content_lines.append(f"**Tool Results:**\n{d['tool_results']}")
        if d.get("decisions"):
            content_lines.append(f"**Decisions:**\n{d['decisions']}")
        if d.get("narrative"):
            content_lines.append(f"**Narrative:**\n{d['narrative']}")
        if not content_lines:
            # Fallback to raw text or a placeholder
            raw = d.get("raw_text", "")
            if raw:
                content_lines.append(raw)
            else:
                content_lines.append("[No structured summary]")
        content = "\n\n".join(content_lines)
        comp_timestamp = d.get("timestamp", "")
        if comp_timestamp:
            content = f"[Compression timestamp: {comp_timestamp}]\n" + content
        return MemoryEntry(
            role="assistant",
            content=content,
            time=comp_timestamp,
            layer=Layer.COMPRESSED,
        )

    def _entry_to_dict(self, entry: MemoryEntry) -> dict:
        """Serialize MemoryEntry to dict. Preserves tool_calls for crash recovery."""
        d = {
            "role": entry.role,
            "content": entry.content,
            "time": entry.time,
        }
        if entry.reasoning:
            d["reasoning_content"] = entry.reasoning
        if entry.tool_name:
            d["name"] = entry.tool_name
        if entry.tool_call_id:
            d["tool_call_id"] = entry.tool_call_id
        if entry.tool_calls:
            d["tool_calls"] = entry.tool_calls
        return d

    def _entry_to_openai_dict(self, entry: MemoryEntry) -> dict:
        """Convert to OpenAI-compatible message dict.

        Critical: OpenAI/DeepSeek API requires assistant messages to have
        non-empty `content` OR non-empty `tool_calls`. Empty string ""
        is treated as missing. This method ensures validity.
        """
        d = {"role": entry.role}

        raw_content = entry.content
        has_tool_calls = bool(entry.tool_calls)

        if raw_content is not None and raw_content != "":
            d["content"] = raw_content
        elif has_tool_calls:
            # content can be empty/null when tool_calls present
            d["content"] = ""
        elif entry.reasoning:
            # Fallback: first line of reasoning as content
            first_line = entry.reasoning.split('\n')[0][:200]
            d["content"] = f"[Thought: {first_line}]"
        else:
            d["content"] = "[No content]"

        if entry.reasoning:
            d["reasoning_content"] = entry.reasoning
        if entry.tool_calls:
            d["tool_calls"] = entry.tool_calls
        if entry.tool_name:
            d["name"] = entry.tool_name
        if entry.tool_call_id:
            d["tool_call_id"] = entry.tool_call_id
        return d

    def _check_overflow(self) -> None:
        """Check token budget and set overflow flag.
        Trigger compression at 70% of max_tokens OR >=10 entries older than sliding window.
        """
        est_tokens = self.get_context_length()
        overflow_token = est_tokens > self.max_tokens * 0.7
        old_entries = len(self._all_entries) - self.window_size
        overflow_old = (old_entries >= 10)
        if overflow_token or overflow_old:
            self.overflow = True
            logger.warning(
                f"Context at {est_tokens} tokens ({est_tokens/self.max_tokens:.0%}), "
                f"old entries {old_entries} — overflow"
            )
        else:
            self.overflow = False

    def _auto_mask(self) -> None:
        """Automatically mask tool outputs beyond sliding window."""
        # Masking happens lazily via masked_entries property; nothing to do eagerly.
        # Kept for future proactive masking if needed.
        pass

    def _periodic_save(self) -> None:
        """Emergency save every 5 messages."""
        if self._save_counter % 5 == 0:
            self._emergency_save()

    def _save_to_file(self, entry: MemoryEntry) -> None:
        """Append to human-readable history file."""
        history_file = self._data_dir / "message_history.md"
        try:
            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(f"**{entry.role}:** [{entry.time}]\n")
                if entry.reasoning:
                    f.write(f"*Reasoning:* {entry.reasoning}\n\n")
                f.write(f"{entry.content or ''}\n---\n\n")
        except Exception as e:
            logger.error(f"History save failed: {e}")

    def _token_count_entry(self, entry: MemoryEntry) -> int:
        """Return cached token count for a MemoryEntry."""
        eid = id(entry)
        if eid not in self._token_counts:
            text = (entry.content or "") + (entry.reasoning or "")
            self._token_counts[eid] = _token_count(text)
        return self._token_counts[eid]

    def log_state(self, tag: str = "") -> None:
        logger.info(f"=== Context state {tag} ===")
        logger.info(f"  Total entries: {len(self._all_entries)}")
        logger.info(f"  Sliding window (last {self.window_size}): {len(self.sliding_window)} entries")
        if self.sliding_window:
            first = self.sliding_window[0]
            last = self.sliding_window[-1]
            first_preview = (first.content[:50] if first.content else "[No content]")
            last_preview = (last.content[:50] if last.content else "[No content]")
            logger.info(f"    first: {first.role} '{first_preview}'")
            logger.info(f"    last:  {last.role} '{last_preview}'")
        logger.info(f"  Compressed entries (dicts): {len(self._compressed_dicts)}")
        total_comp_tokens = sum(
            _token_count((d.get("facts","") + d.get("tool_results","") + d.get("decisions","") + d.get("narrative","")))
            for d in self._compressed_dicts
        )
        logger.info(f"    tokens: ~{total_comp_tokens}")
        logger.info(f"  Masked entries (old): {len(self.masked_entries)}")
        logger.info(f"  Persistent facts: {len(self._persistent.facts)}")
        logger.info(f"  Estimated tokens: {self.get_context_length()}")
        logger.info(f"  Overflow flag: {self.overflow}")
        logger.info("=====================================")
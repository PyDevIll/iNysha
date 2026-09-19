"""File system tools for MASTERMIND v2 — async, registry-integrated."""

import os
import stat
import fnmatch
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

_current_dir: Path = Path.cwd()


def _safe_path(path: str) -> Path:
    """Resolve path relative to agent's virtual cwd."""
    p = Path(path)
    if not p.is_absolute():
        p = _current_dir / p
    return p.resolve()


def human_size(size_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _is_binary(filepath: str, sample_size: int = 1024) -> bool:
    try:
        with open(filepath, 'rb') as f:
            data = f.read(sample_size)
    except Exception:
        return False

    if not data:
        return False

    # Null bytes are almost always a sign of binary data
    if b'\x00' in data:
        return True

    # If the sample can be decoded as UTF-8, it's text (including Cyrillic)
    try:
        data.decode('utf-8')
        return False
    except UnicodeDecodeError:
        pass

    # Fallback heuristic for other encodings or mixed content
    printable = set(range(0x20, 0x7f)) | {0x09, 0x0a, 0x0d}
    non_printable = sum(1 for b in data if b not in printable)
    return (non_printable / len(data)) > 0.30


async def fs_tree(path: str = ".", depth: int = 2, ascii_mode: bool = False) -> str:
    """Recursive directory listing with sizes and dates."""
    p = _safe_path(path)
    if not p.exists():
        return f"Path not found: {path}"
    lines = []

    def _tree(current: Path, d: int, prefix: str) -> None:
        if d < 0:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            lines.append(f"{prefix}Permission denied: {current}")
            return
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "`-- " if (ascii_mode and is_last) else ("|-- " if ascii_mode else ("\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "))
            size_str = human_size(entry.stat().st_size) if entry.is_file() else "0.0 B"
            mod_str = datetime.fromtimestamp(entry.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"{prefix}{connector}{entry.name}  {size_str}  {mod_str}")
            if entry.is_dir() and not entry.name.startswith('.'):
                sub_prefix = prefix + ("    " if is_last else ("    " if ascii_mode else "\u2502   "))
                _tree(entry, d - 1, sub_prefix)

    _tree(p, min(depth, 5), "")
    return '\n'.join(lines)


async def fs_read(file: str, lines: int = 30, start: int = 1) -> str:
    """Read file content with binary detection and size-aware truncation."""
    p = _safe_path(file)
    if not p.exists():
        return f"File not found: {file}"
    if lines > 500:
        lines = 500
    try:
        size = p.stat().st_size
    except OSError as e:
        return f"Error: {e}"
    if _is_binary(str(p)):
        return f"Binary file detected ({human_size(size)})"
    try:
        content = p.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return f"Error reading file: {e}"
    all_lines = content.split('\n')
    total = len(all_lines)
    if size > 128 * 1024:
        first = all_lines[:20]
        last = all_lines[-10:] if total >= 10 else []
        out = [f"File is {human_size(size)} (>128KB, было 50). First 20 + last 10 lines:"]
        for i, line in enumerate(first, 1):
            out.append(f"{i:>6}: {line}")
        if total > 30:
            out.append(f"... ({total - 30} lines omitted)")
        for i, line in enumerate(last, max(1, total - 9)):
            out.append(f"{i:>6}: {line}")
        return '\n'.join(out)
    start_idx = max(0, start - 1)
    end_idx = min(total, start_idx + lines)
    out = []
    for i in range(start_idx, end_idx):
        out.append(f"{i+1:>6}: {all_lines[i]}")
    if end_idx < total:
        out.append(f"... ({total - end_idx} more lines)")
    return '\n'.join(out)


async def fs_tail(file: str, lines: int = 10) -> str:
    """Read last N lines of a text file (like tail)."""
    p = _safe_path(file)
    if not p.exists():
        return f"File not found: {file}"
    try:
        size = p.stat().st_size
    except OSError as e:
        return f"Error: {e}"
    if _is_binary(str(p)):
        return f"Binary file detected ({human_size(size)})"
    try:
        content = p.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return f"Error reading file: {e}"
    all_lines = content.split('\n')
    total = len(all_lines)
    n = min(lines, 200)
    start_idx = max(0, total - n)
    out_lines = []
    for i in range(start_idx, total):
        out_lines.append(f"{i+1:>6}: {all_lines[i]}")
    # Show file size info
    out_lines.insert(0, f"File: {p}  Size: {human_size(size)}  Total lines: {total}")
    if start_idx > 0:
        out_lines.insert(1, f"... ({start_idx} lines omitted)")
    return '\n'.join(out_lines)


async def fs_stat(path: str) -> str:
    """File/directory metadata."""
    p = _safe_path(path)
    if not p.exists():
        return f"Not found: {path}"
    st = p.stat()
    ftype = "directory" if p.is_dir() else "file"
    lines = [
        f"Path: {p}",
        f"Type: {ftype}",
        f"Size: {human_size(st.st_size)}",
        f"Modified: {datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Permissions: {stat.filemode(st.st_mode)}",
    ]
    if not p.is_dir():
        try:
            line_count = p.read_text(encoding='utf-8', errors='replace').count('\n')
            lines.append(f"Lines: {line_count}")
        except Exception:
            pass
    return '\n'.join(lines)



async def fs_grep(
    pattern: str,
    path: str = ".",
    ext: str = "",
    max_files: int = 500,
    regex: bool = False,
    case_sensitive: bool = False,
    max_size: int = 10_000_000,
    force_text: bool = False,
) -> str:
    """
    Recursive text search with advanced options. Skips binary/large files by default.

    Args:
        pattern: Search string or regex pattern (if regex=True).
        path: Directory or file to search (default: .).
        ext: File extension filter (e.g., ".py").
        max_files: Maximum number of files to scan (default: 500).
        regex: If True, treat pattern as a regular expression.
        case_sensitive: If True, perform case-sensitive search.
        max_size: Maximum file size in bytes to read (0 = unlimited, default: 10MB, было 1мб).
        force_text: If True, skip binary detection and read the file as text.

    Returns:
        String with matching lines (file:line:content) or error/summary.
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Path not found: {path}"

    # ── Single‑file mode ────────────────────────────────────────────────
    if p.is_file():
        return await _grep_single_file(
            p, pattern, regex, case_sensitive, max_size, force_text
        )

    # ── Directory scan ──────────────────────────────────────────────────
    pattern_lower = pattern.lower() if not regex else None
    try:
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            pat = re.compile(pattern, flags)
        else:
            pat = None
    except re.error as e:
        return f"Invalid regex: {e}"

    matches: list[str] = []
    skipped_binary: list[str] = []
    skipped_large: list[str] = []
    MAX_MATCHES = 100
    files_scanned = 0

    # Collect files (fast)
    files_to_search: list[Path] = []
    for fp in p.rglob("*"):
        if fp.is_file():
            if ext and not fp.name.endswith(ext):
                continue
            files_to_search.append(fp)
            if len(files_to_search) >= max_files:
                break

    # Process each file (non‑blocking with thread pool)
    def search_file(fp: Path) -> list[str]:
        local_matches: list[str] = []
        try:
            fsize = fp.stat().st_size
        except OSError:
            return local_matches

        # Size check
        if max_size and fsize > max_size:
            return local_matches  # counted by caller

        # Binary detection (unless forced)
        if not force_text and _is_binary(str(fp)):
            return local_matches  # counted by caller

        # Stream line by line to avoid memory blow
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace' if force_text else 'strict') as f:
                for lineno, line in enumerate(f, 1):
                    line = line.rstrip('\n')
                    if regex:
                        if pat.search(line):
                            local_matches.append(f"{fp}:{lineno}:{line}")
                    else:
                        needle = pattern if case_sensitive else pattern_lower
                        haystack = line if case_sensitive else line.lower()
                        if needle in haystack:
                            local_matches.append(f"{fp}:{lineno}:{line}")
                    if len(local_matches) >= MAX_MATCHES:
                        break
        except (UnicodeDecodeError, PermissionError, OSError):
            # If we can't read, treat as binary / inaccessible
            pass
        return local_matches

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for fp in files_to_search:
            futures.append(loop.run_in_executor(pool, search_file, fp))

        for future in asyncio.as_completed(futures):
            try:
                result = await future
                matches.extend(result)
            except Exception:
                pass

    # Count skipped files by size / binary after the fact
    for fp in files_to_search:
        try:
            if max_size and fp.stat().st_size > max_size:
                skipped_large.append(str(fp))
            elif not force_text and _is_binary(str(fp)):
                skipped_binary.append(str(fp))
        except OSError:
            pass

    # Build response
    skipped_msg = ""
    if skipped_binary or skipped_large:
        parts = []
        if skipped_binary:
            c = len(skipped_binary)
            file_list = skipped_binary[:5]
            part = f"{c} binary"
            if file_list:
                part += f" ({', '.join(file_list)})"
            parts.append(part)
        if skipped_large:
            c = len(skipped_large)
            file_list = skipped_large[:5]
            part = f"{c} large (> {human_size(max_size) if max_size else '∞'})"
            if file_list:
                part += f" ({', '.join(file_list)})"
            parts.append(part)
        skipped_msg = f" (skipped: {'; '.join(parts)})"
        # Add tip if more than 5 total skipped files
        if len(skipped_binary) + len(skipped_large) > 5:
            skipped_msg += "\nTip: use force_text=True or max_size=0, or grep specific files with path= to avoid skips"

    if not matches:
        return f"No matches for '{pattern}' in {path}{skipped_msg}"

    out = "\n".join(matches[:MAX_MATCHES])
    if len(matches) > MAX_MATCHES:
        out += f"\n\n... truncated to {MAX_MATCHES} matches"
    if skipped_msg:
        out += f"\n{skipped_msg}"
    return out


async def _grep_single_file(
    fp: Path,
    pattern: str,
    regex: bool,
    case_sensitive: bool,
    max_size: int,
    force_text: bool,
) -> str:
    """Search a single file with the same options."""
    try:
        fsize = fp.stat().st_size
    except OSError as e:
        return f"Error reading file: {e}"

    if max_size and fsize > max_size and not force_text:
        return (f"File {fp} exceeds size limit ({human_size(fsize)} > {human_size(max_size)}). "
                "Use force_text=True or increase max_size.")

    if not force_text and _is_binary(str(fp)):
        return f"Binary file detected ({human_size(fsize)}). Use force_text=True to search anyway."

    try:
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            pat = re.compile(pattern, flags)
        else:
            pat = None
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = []
    MAX_MATCHES = 100
    try:
        with open(fp, 'r', encoding='utf-8', errors='replace' if force_text else 'strict') as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip('\n')
                if regex:
                    if pat.search(line):
                        matches.append(f"{fp}:{lineno}:{line}")
                else:
                    needle = pattern if case_sensitive else pattern.lower()
                    haystack = line if case_sensitive else line.lower()
                    if needle in haystack:
                        matches.append(f"{fp}:{lineno}:{line}")
                if len(matches) >= MAX_MATCHES:
                    break
    except UnicodeDecodeError as e:
        return f"Encoding error reading {fp}: {e}. Try force_text=True."
    except Exception as e:
        return f"Error reading {fp}: {e}"

    if not matches:
        return f"No matches for '{pattern}' in {fp}"
    out = "\n".join(matches)
    if len(matches) >= MAX_MATCHES:
        out += f"\n\n... truncated to {MAX_MATCHES} matches"
    return out


async def fs_find(path: str = ".", name: str = "*") -> str:
    """Find files by case-insensitive glob pattern."""
    p = _safe_path(path)
    if not p.exists():
        return f"Path not found: {path}"
    pattern_lower = name.lower()
    matches = []
    for fp in p.rglob("*"):
        if fnmatch.fnmatch(fp.name.lower(), pattern_lower):
            matches.append(str(fp))
    if not matches:
        return f"No matches for '{name}' in {path}"
    return '\n'.join(matches[:100])


async def fs_mkdir(path: str) -> str:
    """Create directory recursively."""
    p = _safe_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"Created directory: {p}"


async def fs_touch(path: str) -> str:
    """Create empty file or update mtime."""
    p = _safe_path(path)
    if p.exists():
        p.touch()
        return f"Updated mtime: {p}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return f"Created empty file: {p}"


async def fs_rm(path: str, recursive: bool = False) -> str:
    """Delete file or directory."""
    p = _safe_path(path)
    if not p.exists():
        return f"Not found: {path}"
    if p.is_dir():
        if recursive:
            shutil.rmtree(str(p))
            return f"Removed directory: {p}"
        return f"Cannot remove directory without recursive=True: {p}"
    p.unlink()
    return f"Removed file: {p}"


async def fs_mv(src: str, dst: str) -> str:
    """Move/rename file or directory."""
    sp = _safe_path(src)
    dp = _safe_path(dst)
    if not sp.exists():
        return f"Source not found: {src}"
    dp.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(sp), str(dp))
    return f"Moved: {sp} -> {dp}"


async def fs_cp(src: str, dst: str, recursive: bool = False) -> str:
    """Copy file or directory."""
    sp = _safe_path(src)
    dp = _safe_path(dst)
    if not sp.exists():
        return f"Source not found: {src}"
    dp.parent.mkdir(parents=True, exist_ok=True)
    if sp.is_dir():
        if recursive:
            shutil.copytree(str(sp), str(dp), dirs_exist_ok=True)
            return f"Copied directory: {sp} -> {dp}"
        return f"Cannot copy directory without recursive=True: {sp}"
    shutil.copy2(str(sp), str(dp))
    return f"Copied file: {sp} -> {dp}"


async def fs_cd(path: str) -> str:
    """Change agent's virtual working directory."""
    global _current_dir
    p = _safe_path(path)
    if not p.exists():
        return f"Path not found: {path}"
    if not p.is_dir():
        return f"Not a directory: {path}"
    _current_dir = p
    return f"Changed directory to: {p}"


async def fs_pwd() -> str:
    """Print agent's virtual working directory."""
    return str(_current_dir)


async def fs_sizes(path: str = ".", top: int = 20) -> str:
    """List largest files in directory."""
    p = _safe_path(path)
    if not p.exists():
        return f"Path not found: {path}"
    entries = []
    for fp in p.rglob("*"):
        if fp.is_file():
            try:
                entries.append((fp.stat().st_size, str(fp)))
            except OSError:
                pass
    entries.sort(key=lambda x: -x[0])
    lines = []
    for sz, fp in entries[:top]:
        lines.append(f"{human_size(sz):>10}  {fp}")
    if len(entries) > top:
        lines.append(f"... ({len(entries) - top} more files)")
    return '\n'.join(lines)


async def fs_edit(file: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines in a text file. Line numbers are 1-indexed, inclusive."""
    p = _safe_path(file)
    if not p.exists():
        return f"File not found: {file}"
    if _is_binary(str(p)):
        return "Cannot edit binary file"
    try:
        content = p.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file: {e}"
    lines_list = content.split('\n')
    if start_line < 1:
        start_line = 1
    if end_line > len(lines_list):
        end_line = len(lines_list)
    if start_line > end_line:
        return f"Invalid range: {start_line}-{end_line}"
    new_lines = new_content.split('\n')
    result = lines_list[:start_line - 1] + new_lines + lines_list[end_line:]
    p.write_text('\n'.join(result), encoding='utf-8')
    return f"Edited {file}: replaced lines {start_line}-{end_line} ({len(new_lines)} lines inserted)"


async def fs_append(file: str, content: str) -> str:
    """Append text to end of file."""
    p = _safe_path(file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf-8') as f:
        f.write(content)
    return f"Appended to {file}"


async def fs_read_docx(file: str, max_chars: int = 20000) -> str:
    """Extract text from a .docx (Word) document using docxpy."""
    p = _safe_path(file)
    if not p.exists():
        return f"File not found: {file}"
    if p.suffix.lower() != '.docx':
        return f"Not a .docx file: {file}"
    try:
        import docxpy
    except ImportError:
        return "docxpy is not installed. Run: pip install docxpy"

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, docxpy.process, str(p))
    except Exception as e:
        return f"Error reading docx: {e}"

    if text is None:
        text = ""
    text = text.strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated, total {len(text)} chars]"
    return text


TOOL_DEFINITIONS = [
    ("fs_tree", fs_tree, "Recursive directory listing with sizes and dates", {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path (default: .)"},
            "depth": {"type": "integer", "description": "Depth level (default 2, max 5)"},
            "ascii_mode": {"type": "boolean", "description": "Use ASCII characters for tree"},
        },
    }),
    ("fs_read", fs_read, "Read file content with binary detection and size-aware truncation", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "File path"},
            "lines": {"type": "integer", "description": "Number of lines (default 30, max 80)"},
            "start": {"type": "integer", "description": "Starting line (1-indexed)"},
        },
        "required": ["file"],
    }),
    ("fs_tail", fs_tail, "Read last N lines of a text file (like tail)", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "File path"},
            "lines": {"type": "integer", "description": "Number of lines to read from the end (default: 10, max: 200)"},
        },
        "required": ["file"],
    }),
    ("fs_stat", fs_stat, "File/directory metadata", {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path"}},
        "required": ["path"],
    }),
    ("fs_grep", fs_grep, "Recursive case‑insensitive text search with regex and size limits", {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (string or regex if regex=True)"},
            "path": {"type": "string", "description": "Directory or file to search (default: .)"},
            "ext": {"type": "string", "description": "File extension filter (e.g., .py)"},
            "max_files": {"type": "integer", "description": "Max files to scan (default: 500)"},
            "regex": {"type": "boolean", "description": "Treat pattern as regex (default: false)"},
            "case_sensitive": {"type": "boolean", "description": "Case‑sensitive search (default: false)"},
            "max_size": {"type": "integer", "description": "Max file size in bytes (0 = unlimited, default: 1MB)"},
            "force_text": {"type": "boolean", "description": "Skip binary detection (default: false)"},
        },
        "required": ["pattern"],
    }),
    ("fs_find", fs_find, "Find files by case-insensitive glob pattern", {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to search (default: .)"},
            "name": {"type": "string", "description": "Glob pattern (default: *)"},
        },
    }),
    ("fs_mkdir", fs_mkdir, "Create directory recursively", {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path"}},
        "required": ["path"],
    }),
    ("fs_touch", fs_touch, "Create empty file or update mtime", {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"],
    }),
    ("fs_rm", fs_rm, "Delete file or directory", {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to remove"},
            "recursive": {"type": "boolean", "description": "Remove directories recursively"},
        },
        "required": ["path"],
    }),
    ("fs_mv", fs_mv, "Move/rename file or directory", {
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "Source path"},
            "dst": {"type": "string", "description": "Destination path"},
        },
        "required": ["src", "dst"],
    }),
    ("fs_cp", fs_cp, "Copy file or directory", {
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "Source path"},
            "dst": {"type": "string", "description": "Destination path"},
            "recursive": {"type": "boolean", "description": "Copy directories recursively"},
        },
        "required": ["src", "dst"],
    }),
    ("fs_cd", fs_cd, "Change agent's virtual working directory", {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path"}},
        "required": ["path"],
    }),
    ("fs_pwd", fs_pwd, "Print agent's virtual working directory", {
        "type": "object",
        "properties": {},
    }),
    ("fs_sizes", fs_sizes, "List largest files in directory", {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to scan (default: .)"},
            "top": {"type": "integer", "description": "Number of entries (default 20)"},
        },
    }),
    ("fs_append", fs_append, "Append text to end of file", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Content to append"},
        },
        "required": ["file", "content"],
    }),
    ("fs_read_docx", fs_read_docx, "Extract text from a .docx (Word) document using docxpy", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "Path to the .docx file"},
            "max_chars": {"type": "integer", "description": "Max characters to return (default: 20000)"},
        },
        "required": ["file"],
    }),
]


def register_all(registry):
    """Register all file system tools with the given registry."""
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} FS tools")

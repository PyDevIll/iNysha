"""Advanced file-editing tools for MASTERMIND v2 — Aider-inspired layered matching.

Formats supported:
  - SEARCH/REPLACE blocks (EditBlock, Aider's "diff" format)
  - Unified diff (Aider's "udiff" format)
  - Whole-file rewrite

Layered matching strategies (tried in order):
  1. Exact match
  2. Whitespace-uniform-outdent (LLM whitespace errors)
  3. Skip-blank-leading-line
  4. Elision handling (... marker)
  5. Fuzzy match via difflib.SequenceMatcher
"""

from __future__ import annotations

import difflib
import math
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from .fs_tools import _safe_path, _is_binary


# ═══════════════════════════════════════════════════════════════════════════════
# Core matching engine — layered strategy cascade
# ═══════════════════════════════════════════════════════════════════════════════

def _prep(text: str) -> tuple[str, list[str]]:
    """Normalize text and split into lines (keeping line endings)."""
    if text and not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)
    return text, lines


def _perfect_replace(whole_lines: list[str], part_lines: list[str],
                     replace_lines: list[str]) -> Optional[str]:
    """Exact match: find part_lines in whole_lines and replace."""
    part_tup = tuple(part_lines)
    part_len = len(part_lines)
    for i in range(len(whole_lines) - part_len + 1):
        if tuple(whole_lines[i:i + part_len]) == part_tup:
            return "".join(whole_lines[:i] + replace_lines + whole_lines[i + part_len:])
    return None


def _match_but_for_leading_whitespace(whole_lines: list[str],
                                       part_lines: list[str]) -> Optional[str]:
    """Check if lines match except for uniform leading whitespace.
    Returns the whitespace prefix to add to replacement lines, or None."""
    num = len(whole_lines)
    if not all(whole_lines[i].lstrip() == part_lines[i].lstrip() for i in range(num)):
        return None
    add = {whole_lines[i][:len(whole_lines[i]) - len(part_lines[i])]
           for i in range(num) if whole_lines[i].strip()}
    if len(add) != 1:
        return None
    return add.pop()


def _replace_with_missing_leading_whitespace(
    whole_lines: list[str], part_lines: list[str],
    replace_lines: list[str]) -> Optional[str]:
    """Handle LLM uniformly outdenting SEARCH/REPLACE blocks."""
    # Compute minimum uniform leading whitespace to strip
    leading = [len(p) - len(p.lstrip()) for p in part_lines if p.strip()] + \
              [len(r) - len(r.lstrip()) for r in replace_lines if r.strip()]
    if not leading or min(leading) == 0:
        return None

    num_leading = min(leading)
    stripped_part = [p[num_leading:] if p.strip() else p for p in part_lines]
    stripped_replace = [r[num_leading:] if r.strip() else r for r in replace_lines]

    num_part = len(stripped_part)
    for i in range(len(whole_lines) - num_part + 1):
        add_leading = _match_but_for_leading_whitespace(
            whole_lines[i:i + num_part], stripped_part)
        if add_leading is None:
            continue
        adjusted_replace = [add_leading + r if r.strip() else r for r in stripped_replace]
        return "".join(whole_lines[:i] + adjusted_replace + whole_lines[i + num_part:])
    return None


def _perfect_or_whitespace(whole_lines: list[str], part_lines: list[str],
                            replace_lines: list[str]) -> Optional[str]:
    """Try exact match first, then whitespace-flexible."""
    res = _perfect_replace(whole_lines, part_lines, replace_lines)
    if res:
        return res
    return _replace_with_missing_leading_whitespace(
        whole_lines, part_lines, replace_lines)


def _try_dotdotdots(whole: str, part: str, replace: str) -> Optional[str]:
    """Handle Aider-style ... elision in SEARCH/REPLACE blocks."""
    dots_re = re.compile(r"(^\s*\.\.\.\n)", re.MULTILINE | re.DOTALL)
    part_pieces = re.split(dots_re, part)
    replace_pieces = re.split(dots_re, replace)

    if len(part_pieces) != len(replace_pieces):
        raise ValueError("Unpaired ... in SEARCH/REPLACE block")
    if len(part_pieces) == 1:
        return None  # no dots

    # All odd-indexed pieces must be identical ... markers
    for i in range(1, len(part_pieces), 2):
        if part_pieces[i] != replace_pieces[i]:
            raise ValueError("Unmatched ... in SEARCH/REPLACE block")

    # Process even-indexed pieces (actual text)
    for i in range(0, len(part_pieces), 2):
        p = part_pieces[i]
        r = replace_pieces[i]
        if not p and not r:
            continue
        if not p and r:
            if not whole.endswith("\n"):
                whole += "\n"
            whole += r
            continue
        count = whole.count(p)
        if count == 0:
            raise ValueError(f"Elision piece not found: {p[:50]!r}...")
        if count > 1:
            raise ValueError(f"Elision piece not unique ({count} occurrences)")
        whole = whole.replace(p, r, 1)

    return whole


def _replace_closest_edit_distance(whole_lines: list[str], part: str,
                                     part_lines: list[str],
                                     replace_lines: list[str]) -> Optional[str]:
    """Fuzzy match: find most similar chunk via difflib.SequenceMatcher."""
    similarity_thresh = 0.80
    max_similarity = 0.0
    best_start = -1
    best_end = -1
    scale = 0.15
    min_len = math.floor(len(part_lines) * (1 - scale))
    max_len = math.ceil(len(part_lines) * (1 + scale))

    for length in range(max(min_len, 1), min(max_len, len(whole_lines)) + 1):
        for i in range(len(whole_lines) - length + 1):
            chunk = "".join(whole_lines[i:i + length])
            similarity = difflib.SequenceMatcher(None, chunk, part).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                best_start = i
                best_end = i + length

    if max_similarity < similarity_thresh:
        return None

    return "".join(
        whole_lines[:best_start] + replace_lines + whole_lines[best_end:])


def replace_most_similar_chunk(content: str, search: str, replace: str) -> Optional[str]:
    """Core engine: find `search` in `content` and replace with `replace`.

    Tries strategies in order:
      1. Exact match
      2. Whitespace-flexible (uniform outdent)
      3. Skip blank leading line
      4. ... elision handling
      5. Fuzzy difflib match
    """
    whole, whole_lines = _prep(content)
    part, part_lines = _prep(search)
    repl, repl_lines = _prep(replace)

    # Strategy 1 & 2: exact + whitespace
    res = _perfect_or_whitespace(whole_lines, part_lines, repl_lines)
    if res:
        return res

    # Strategy 3: drop leading empty line (LLMs add these spuriously)
    if len(part_lines) > 2 and not part_lines[0].strip():
        res = _perfect_or_whitespace(whole_lines, part_lines[1:], repl_lines)
        if res:
            return res

    # Strategy 4: ... elision
    try:
        res = _try_dotdotdots(content, search, replace)
        if res:
            return res
    except ValueError:
        pass

    # Strategy 5: fuzzy match
    res = _replace_closest_edit_distance(whole_lines, part, part_lines, repl_lines)
    if res:
        return res

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH/REPLACE block parsing
# ═══════════════════════════════════════════════════════════════════════════════

HEAD_PAT = re.compile(r"^<{5,9} SEARCH\s*$")
DIVIDER_PAT = re.compile(r"^={5,9}\s*$")
UPDATED_PAT = re.compile(r"^>{5,9} REPLACE\s*$")
SEPARATORS_RE = re.compile(
    r"^((?:<{5,9} SEARCH\s*|={5,9}\s*|>{5,9} REPLACE\s*)[ ]*\n)",
    re.MULTILINE | re.DOTALL)

DEFAULT_FENCE = ("```", "```")


def _strip_filename(filename: str) -> Optional[str]:
    """Extract filename from a line, stripping fences and markers."""
    filename = filename.strip()
    if filename == "...":
        return None
    for fence_start in ("```", "````"):
        if filename.startswith(fence_start):
            candidate = filename[len(fence_start):]
            if candidate and ("." in candidate or "/" in candidate or "\\" in candidate):
                return candidate.strip()
            return None
    filename = filename.rstrip(":")
    filename = filename.lstrip("#")
    filename = filename.strip("`*")
    return filename


def _strip_quoted_wrapping(text: str, fname: str = None) -> str:
    """Remove filename prefix and fence wrapping from block content."""
    if not text:
        return text
    lines = text.splitlines()
    # Remove filename line if present
    if fname and lines and Path(fname).name in lines[0]:
        lines = lines[1:]
    # Remove fence wrapping
    if lines and lines[0].startswith("```") and lines[-1].startswith("```"):
        lines = lines[1:-1]
    result = "\n".join(lines)
    if result and result[-1] != "\n":
        result += "\n"
    return result


def find_edit_blocks(content: str) -> list[tuple[Optional[str], str, str]]:
    """Parse SEARCH/REPLACE blocks from LLM output.

    Returns list of (filename, search_text, replace_text) tuples.
    Filename may be None if embedded in a fenced block with file header.
    """
    lines = content.splitlines(keepends=True)
    i = 0
    edits = []
    current_filename = None

    while i < len(lines):
        line = lines[i]

        # Skip non-edit blocks (shell, etc.)
        shell_starts = ("```bash", "```sh", "```shell", "```cmd",
                        "```batch", "```powershell", "```ps1",
                        "```zsh", "```fish", "```ksh", "```csh", "```tcsh")
        next_is_editblock = (
            i + 1 < len(lines) and HEAD_PAT.match(lines[i + 1].strip())) or \
            (i + 2 < len(lines) and HEAD_PAT.match(lines[i + 2].strip()))

        if any(line.strip().startswith(s) for s in shell_starts) and not next_is_editblock:
            # Skip shell block
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            continue

        # Try to extract filename from the line
        stripped = line.strip()
        maybe_fname = _strip_filename(stripped)

        # Check if next line starts SEARCH block
        next_is_search = i + 1 < len(lines) and HEAD_PAT.match(lines[i + 1].strip())
        # Or line after next
        next_next_is_search = i + 2 < len(lines) and HEAD_PAT.match(lines[i + 2].strip())

        if maybe_fname and (next_is_search or next_next_is_search):
            current_filename = maybe_fname
            i += 1
            continue

        # Check for SEARCH block start
        if HEAD_PAT.match(stripped):
            # Collect SEARCH content
            i += 1
            search_lines = []
            while i < len(lines):
                if DIVIDER_PAT.match(lines[i].strip()):
                    break
                search_lines.append(lines[i])
                i += 1
            else:
                # No divider found — malformed
                break

            i += 1  # skip divider
            # Collect REPLACE content
            replace_lines = []
            while i < len(lines):
                if UPDATED_PAT.match(lines[i].strip()):
                    break
                replace_lines.append(lines[i])
                i += 1
            else:
                break

            search_text = "".join(search_lines)
            replace_text = "".join(replace_lines)
            edits.append((current_filename, search_text, replace_text))
            i += 1
            continue

        i += 1

    return edits


# ═══════════════════════════════════════════════════════════════════════════════
# Unified diff parsing and application
# ═══════════════════════════════════════════════════════════════════════════════

def _hunk_to_before_after(hunk: list[str]) -> tuple[str, str]:
    """Split a unified diff hunk into before and after text."""
    before = []
    after = []
    for line in hunk:
        if len(line) < 2:
            op = " "
            text = line
        else:
            op = line[0]
            text = line[1:]
        if op == " ":
            before.append(text)
            after.append(text)
        elif op == "-":
            before.append(text)
        elif op == "+":
            after.append(text)
    return "".join(before), "".join(after)


def _find_diffs(content: str) -> list[tuple[Optional[str], list[str]]]:
    """Parse unified diff blocks from LLM output.

    Returns list of (filename, hunk_lines) tuples.
    """
    if not content.endswith("\n"):
        content += "\n"
    lines = content.splitlines(keepends=True)
    line_num = 0
    all_edits = []

    while line_num < len(lines):
        line = lines[line_num]
        if line.startswith("```diff"):
            line_num, edits = _process_fenced_diff(lines, line_num + 1)
            all_edits += edits
            continue
        line_num += 1

    return all_edits


def _process_fenced_diff(lines: list[str], start: int) -> tuple[int, list]:
    """Process a ```diff fenced block."""
    # Find end of fence
    end = start
    while end < len(lines):
        if lines[end].startswith("```"):
            break
        end += 1

    block = lines[start:end]
    if not block:
        return end + 1, []

    # Add sentinel
    block.append("@@ @@")

    # Extract filename from ---/+++ headers
    fname = None
    if block[0].startswith("--- ") and block[1].startswith("+++ "):
        b_fname = block[1][4:].strip()
        if b_fname.startswith("b/"):
            fname = b_fname[2:]
        elif b_fname == "/dev/null":
            a_fname = block[0][4:].strip()
            if a_fname.startswith("a/"):
                fname = a_fname[2:]
        else:
            fname = b_fname
        block = block[2:]

    edits = []
    keeper = False
    hunk = []
    for line in block:
        hunk.append(line)
        if len(line) < 2:
            continue
        op = line[0]
        if op in "-+":
            keeper = True
            continue
        if op != "@":
            continue
        if not keeper:
            hunk = []
            continue
        hunk = hunk[:-1]  # Remove the @@ line
        edits.append((fname, hunk))
        hunk = []
        keeper = False

    return end + 1, edits


def _apply_hunk(content: str, hunk: list[str]) -> Optional[str]:
    """Apply a single unified diff hunk to content."""
    before, after = _hunk_to_before_after(hunk)
    if not before:
        return None
    if not before.strip():
        return content + after
    return replace_most_similar_chunk(content, before, after)


def _apply_hunk_partial(content: str, hunk: list[str]) -> Optional[str]:
    """Apply a hunk with progressive context reduction."""
    # Separate context from changes
    ops = [line[0] if len(line) >= 2 else " " for line in hunk]
    # Group into sections: context, changes, context, changes, ...
    sections = []
    section = []
    cur_op = " "
    for i, line in enumerate(hunk):
        op = ops[i]
        if op != cur_op:
            sections.append(section)
            section = []
            cur_op = op
        section.append(line)
    sections.append(section)
    if cur_op != " ":
        sections.append([])

    # Try applying each change section with surrounding context
    for i in range(1, len(sections), 2):
        # Index 0,2,4... = context; 1,3,5... = changes
        prec = sections[i - 1] if i - 1 >= 0 else []
        changes = sections[i]
        foll = sections[i + 1] if i + 1 < len(sections) else []

        # Try with decreasing context
        total_context = len(prec) + len(foll)
        for drop in range(total_context + 1):
            use = total_context - drop
            for use_prec in range(len(prec), -1, -1):
                if use_prec > use:
                    continue
                use_foll = use - use_prec
                if use_foll > len(foll):
                    continue
                this_prec = prec[-use_prec:] if use_prec else []
                this_foll = foll[:use_foll]
                trial_hunk = this_prec + changes + this_foll
                res = _apply_hunk(content, trial_hunk)
                if res:
                    return res
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Similar line finder for error messages
# ═══════════════════════════════════════════════════════════════════════════════

def _find_similar_lines(search: str, content: str, max_lines: int = 8) -> str:
    """Find lines in content most similar to the search block."""
    search_lines = search.splitlines()
    content_lines = content.splitlines()
    if not search_lines or not content_lines:
        return ""

    best_ratio = 0.0
    best_start = 0
    search_len = len(search_lines)

    for i in range(len(content_lines) - min(search_len, len(content_lines)) + 1):
        chunk = "\n".join(content_lines[i:i + search_len])
        ratio = difflib.SequenceMatcher(None, chunk, search).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio > 0.3:
        end = min(best_start + max_lines, len(content_lines))
        return "\n".join(content_lines[best_start:end])

    return "\n".join(content_lines[:max_lines])


# ═══════════════════════════════════════════════════════════════════════════════
# Public API: fs_aedit (advanced edit)
# ═══════════════════════════════════════════════════════════════════════════════

async def fs_aedit(
    file: str,
    search: str,
    replace: str,
    dry_run: bool = False,
    fuzzy: bool = True,
    create_if_missing: bool = True,
) -> str:
    """Advanced file editing with layered matching strategies.

    Uses Aider's SEARCH/REPLACE block format with cascading match strategies:
    exact → whitespace-flexible → skip-blank-line → elision → fuzzy.

    Args:
        file: Path to the file to edit
        search: The exact text block to find (or leave empty to create/append)
        replace: The replacement text block
        dry_run: If true, return what would be changed without writing
        fuzzy: Enable fuzzy matching as last-resort strategy
        create_if_missing: Create file if it doesn't exist and search is empty

    Returns:
        Summary of the edit operation.
    """
    p = _safe_path(file)
    file_exists = p.exists()

    if file_exists and _is_binary(str(p)):
        return "ERROR: Cannot edit binary file"

    # Read current content
    if file_exists:
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR reading file: {e}"
    else:
        content = ""

    # Strip wrapping from search/replace
    clean_search = _strip_quoted_wrapping(search, file)
    clean_replace = _strip_quoted_wrapping(replace, file)

    # Case: creating a new file
    if not file_exists and not clean_search.strip():
        if create_if_missing:
            if dry_run:
                return f"[DRY RUN] Would create {file} with {len(clean_replace)} chars"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(clean_replace, encoding="utf-8")
            return f"Created {file} ({len(clean_replace)} chars)"
        else:
            return f"ERROR: File {file} does not exist and create_if_missing=False"

    # Case: appending to file (empty search)
    if not clean_search.strip():
        new_content = content + clean_replace
        if dry_run:
            return f"[DRY RUN] Would append {len(clean_replace)} chars to {file}"
        p.write_text(new_content, encoding="utf-8")
        return f"Appended {len(clean_replace)} chars to {file} ({len(new_content)} total)"

    # Attempt replacement
    if fuzzy:
        new_content = replace_most_similar_chunk(content, clean_search, clean_replace)
    else:
        # Only exact and whitespace-flexible
        whole, whole_lines = _prep(content)
        part, part_lines = _prep(clean_search)
        repl, repl_lines = _prep(clean_replace)
        new_content = _perfect_or_whitespace(whole_lines, part_lines, repl_lines)

    if new_content is not None:
        if dry_run:
            return f"[DRY RUN] Would replace in {file} ({len(clean_search)}→{len(clean_replace)} chars)"
        p.write_text(new_content, encoding="utf-8")
        return f"Edited {file}: replaced {len(clean_search)}→{len(clean_replace)} chars ({len(new_content)} total)"

    # Match failed — build detailed error
    error_parts = [f"SEARCH block failed to match in {file}"]
    similar = _find_similar_lines(clean_search, content)
    if similar:
        error_parts.append(f"\nMost similar lines in {file}:\n```\n{similar}\n```")
    if clean_replace.strip() in content:
        error_parts.append("\nNOTE: REPLACE content already exists in file!")
    error_parts.append("\nThe SEARCH block must match existing content exactly "
                       "(after whitespace normalization).")
    return "\n".join(error_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API: fs_apply_patch (unified diff)
# ═══════════════════════════════════════════════════════════════════════════════

async def fs_apply_patch(
    file: str,
    patch: str,
    dry_run: bool = False,
) -> str:
    """Apply a unified diff patch to a file.

    Parses unified diff format with ---/+++ headers and @@ hunks.
    Supports multiple hunks per patch, progressive context reduction.

    Args:
        file: Path to the file to patch
        patch: Unified diff content (---/+++/@@ format)
        dry_run: If true, return what would be changed without writing

    Returns:
        Summary of patch application.
    """
    p = _safe_path(file)

    if p.exists() and _is_binary(str(p)):
        return "ERROR: Cannot patch binary file"

    if p.exists():
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR reading file: {e}"
    else:
        content = ""

    # Parse hunks from patch
    hunks = _find_diffs(patch)

    if not hunks:
        # Try parsing as bare hunks (without ---/+++ headers)
        before, after = _hunk_to_before_after(patch.splitlines(keepends=True))
        if before or after:
            hunks = [(file, patch.splitlines(keepends=True))]
        else:
            return "ERROR: No valid hunks found in patch"

    applied = 0
    failed = 0
    errors = []

    for fname, hunk in hunks:
        target = fname or file
        target_path = _safe_path(target)

        if target_path.exists():
            try:
                target_content = target_path.read_text(encoding="utf-8")
            except Exception as e:
                errors.append(f"ERROR reading {target}: {e}")
                failed += 1
                continue
        else:
            target_content = ""

        # Try direct hunk application
        new_content = _apply_hunk(target_content, hunk)

        # Try partial context reduction
        if new_content is None:
            new_content = _apply_hunk_partial(target_content, hunk)

        if new_content is not None:
            if not dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(new_content, encoding="utf-8")
            content = new_content
            applied += 1
        else:
            before_text, _ = _hunk_to_before_after(hunk)
            similar = _find_similar_lines(before_text, target_content)
            errors.append(
                f"Hunk failed to match in {target}. "
                f"Most similar lines:\n```\n{similar}\n```"
            )
            failed += 1

    if failed and not applied:
        return "ERROR:\n" + "\n\n".join(errors)

    result = f"Patch applied to {file}: {applied}/{applied + failed} hunks succeeded"
    if failed:
        result += f"\n{len(errors)} hunks failed:\n" + "\n".join(errors)
    if dry_run:
        result = "[DRY RUN] " + result
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Public API: fs_write_file (whole-file rewrite)
# ═══════════════════════════════════════════════════════════════════════════════

async def fs_write_file(
    file: str,
    content: str,
    dry_run: bool = False,
) -> str:
    """Write complete file content (whole-file rewrite).

    Args:
        file: Path to the file to write
        content: Complete new file content
        dry_run: If true, return what would be changed without writing

    Returns:
        Summary of the write operation.
    """
    p = _safe_path(file)
    old_size = 0
    if p.exists():
        old_size = len(p.read_text(encoding="utf-8"))

    if dry_run:
        return f"[DRY RUN] Would write {file} ({old_size}→{len(content)} chars)"

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {file} ({old_size}→{len(content)} chars)"


# ═══════════════════════════════════════════════════════════════════════════════
# Public API: fs_edit_blocks (multi-edit from SEARCH/REPLACE blocks)
# ═══════════════════════════════════════════════════════════════════════════════

async def fs_edit_blocks(
    blocks: str,
    dry_run: bool = False,
    fuzzy: bool = True,
) -> str:
    """Parse and apply multiple SEARCH/REPLACE blocks from LLM output.

    The blocks string should contain one or more SEARCH/REPLACE blocks
    in Aider's EditBlock format, optionally preceded by filenames:

        path/to/file.py
        <<<<<<< SEARCH
        old code
        =======
        new code
        >>>>>>> REPLACE

    Args:
        blocks: String containing one or more SEARCH/REPLACE blocks
        dry_run: If true, return what would be changed without writing
        fuzzy: Enable fuzzy matching

    Returns:
        Summary of all edit operations.
    """
    edits = find_edit_blocks(blocks)

    if not edits:
        return ("ERROR: No valid SEARCH/REPLACE blocks found. "
                "Use format:\n"
                "  path/to/file\n"
                "  <<<<<<< SEARCH\n  old_code\n  =======\n  new_code\n  >>>>>>> REPLACE")

    results = []
    succeeded = 0
    failed = 0

    for fname, search_text, replace_text in edits:
        if not fname:
            results.append("ERROR: Missing filename for SEARCH/REPLACE block")
            failed += 1
            continue

        result = await fs_aedit(
            file=fname,
            search=search_text,
            replace=replace_text,
            dry_run=dry_run,
            fuzzy=fuzzy,
        )

        if result.startswith("ERROR"):
            failed += 1
        else:
            succeeded += 1
        results.append(f"  {fname}: {result}")

    header = f"Applied {succeeded}/{len(edits)} edit blocks"
    if dry_run:
        header = "[DRY RUN] " + header
    return header + "\n" + "\n".join(results)


# ═══════════════════════════════════════════════════════════════════════════════
# Line-number based diff tools (new)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_unified_diff_v2(diff_text: str) -> list[dict]:
    """Parse unified diff text and extract hunks (line‑number version).

    Returns a list of dicts, each containing:
        old_start, old_count, new_start, new_count,
        context_lines, remove_lines, add_lines,
        body_lines (raw lines after @@), new_block_lines (result lines).
    """
    lines = diff_text.splitlines(keepends=True)
    hunks = []
    hunk_lines = []
    in_hunk = False

    for line in lines:
        if line.startswith('@@'):
            if in_hunk and hunk_lines:
                data = _extract_hunk_data(hunk_lines)
                if data:
                    hunks.append(data)
                hunk_lines = []
            in_hunk = True
            hunk_lines.append(line)
        elif in_hunk:
            hunk_lines.append(line)

    if in_hunk and hunk_lines:
        data = _extract_hunk_data(hunk_lines)
        if data:
            hunks.append(data)

    return hunks


def _extract_hunk_data(hunk_lines: list[str]) -> Optional[dict]:
    """Extract structured data from a single hunk (list of lines)."""
    if not hunk_lines:
        return None
    first = hunk_lines[0]
    m = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', first)
    if not m:
        return None
    old_start = int(m.group(1))
    old_count = int(m.group(2)) if m.group(2) else 1
    new_start = int(m.group(3))
    new_count = int(m.group(4)) if m.group(4) else 1

    body = hunk_lines[1:]   # lines after @@ (with status prefix)
    context = []
    remove = []
    add = []
    for line in body:
        if not line:
            continue
        op = line[0]
        text = line[1:]   # keep trailing newline
        if op == ' ':
            context.append(text)
        elif op == '-':
            remove.append(text)
        elif op == '+':
            add.append(text)

    # Lines that should appear in the result for this hunk (context + additions)
    new_block_lines = [line[1:] for line in body if line and line[0] in (' ', '+')]

    return {
        'old_start': old_start,
        'old_count': old_count,
        'new_start': new_start,
        'new_count': new_count,
        'context_lines': context,
        'remove_lines': remove,
        'add_lines': add,
        'body_lines': body,            # keep raw lines for exact matching
        'new_block_lines': new_block_lines,
    }


def _fuzzy_find_hunk(content_lines: list[str], hunk: dict) -> Optional[int]:
    """Fuzzy‑search for the hunk's old content in content_lines.

    Returns the 0‑based start index if a sufficiently similar block is found,
    otherwise None.
    """
    # Reconstruct the lines that should be present in the old file
    old_lines = []
    for line in hunk['body_lines']:
        if line and line[0] in (' ', '-'):
            old_lines.append(line[1:].rstrip('\n'))
    if not old_lines:
        return None

    target = '\n'.join(old_lines)
    n = len(old_lines)
    best_ratio = 0.0
    best_start = -1

    for i in range(len(content_lines) - n + 1):
        chunk = '\n'.join(content_lines[i:i + n])
        ratio = difflib.SequenceMatcher(None, chunk, target).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio >= 0.7:
        return best_start
    return None


def _apply_hunk_lines(content_lines: list[str], hunk: dict) -> Optional[list[str]]:
    """Apply a single hunk to the content line list.

    Tries exact match first (position‑based), then fuzzy fallback.
    Returns the new list of lines, or None on failure.
    """
    new_lines = list(content_lines)
    # ----- exact match using line‑by‑line comparison -----
    idx = hunk['old_start'] - 1   # 0‑based
    body = hunk['body_lines']
    # Check that we have enough lines in the file from idx
    # We'll try to match each body line starting at idx.
    # We'll use a copy for scanning, but if success we know the region.
    cur = idx
    ok = True
    for line in body:
        if not line:
            continue
        op = line[0]
        text = line[1:]
        if cur >= len(content_lines):
            ok = False
            break
        file_line = content_lines[cur]
        if op == ' ':
            if file_line.rstrip('\n') != text.rstrip('\n'):
                ok = False
                break
            cur += 1
        elif op == '-':
            if file_line.rstrip('\n') != text.rstrip('\n'):
                ok = False
                break
            cur += 1
        elif op == '+':
            # does not consume a file line
            pass

    if ok:
        # Replace region [idx, cur) with new_block_lines
        new_lines[idx:cur] = hunk['new_block_lines']
        return new_lines

    # ----- fuzzy fallback -----
    start = _fuzzy_find_hunk(content_lines, hunk)
    if start is not None:
        # We'll replace the matched region (of length = number of old_lines)
        old_lines_count = sum(1 for line in hunk['body_lines']
                             if line and line[0] in (' ', '-'))
        new_lines[start:start + old_lines_count] = hunk['new_block_lines']
        return new_lines

    return None


async def fs_edit_diff(
    file: str,
    diff: str,
    dry_run: bool = False,
    encoding: str = "utf-8",
) -> str:
    """Apply a unified diff to a file using line‑number based matching.

    Args:
        file: Path to the file to edit.
        diff: Unified diff text (---/+++/@@ format, may omit ---/+++ headers).
        dry_run: If True, preview changes without saving.
        encoding: File encoding (default utf‑8).

    Returns:
        Summary of the edit operation.
    """
    p = _safe_path(file)
    if not p.exists():
        return f"ERROR: File {file} does not exist"
    if _is_binary(str(p)):
        return "ERROR: Cannot edit binary file"

    try:
        content = p.read_text(encoding=encoding)
    except Exception as e:
        return f"ERROR reading file: {e}"

    content_lines = content.splitlines(keepends=True)
    hunks = _parse_unified_diff_v2(diff)
    if not hunks:
        return "ERROR: No valid hunks found in diff"

    for hunk in hunks:
        try:
            new_content_lines = _apply_hunk_lines(content_lines, hunk)
        except Exception as e:
            return f"ERROR applying hunk at line {hunk.get('old_start', '?')}: {e}"
        if new_content_lines is None:
            return (
                f"ERROR: Hunk at old_start={hunk['old_start']} "
                f"failed to match in file"
            )
        content_lines = new_content_lines

    new_content = ''.join(content_lines)

    if dry_run:
        return f"[DRY RUN] Would edit {file}: {len(hunks)} hunk(s) applied"

    try:
        p.write_text(new_content, encoding=encoding)
    except Exception as e:
        return f"ERROR writing file: {e}"

    return f"Edited {file}: applied {len(hunks)} hunk(s)"


async def fs_edit(
    file: str,
    search: str,
    replace: str,
    dry_run: bool = False,
    encoding: str = "utf-8",
) -> str:
    """Simple search‑and‑replace edit (no diff, no Aider format).

    First tries exact (single) replacement, then falls back to fuzzy matching
    (via replace_most_similar_chunk). Preserves file encoding.

    Args:
        file: Path to the file to edit.
        search: Text to find (exact string).
        replace: Replacement text.
        dry_run: If True, preview changes without saving.
        encoding: File encoding (default utf‑8).

    Returns:
        Summary of the edit operation.
    """
    p = _safe_path(file)
    if not p.exists():
        return f"ERROR: File {file} does not exist"
    if _is_binary(str(p)):
        return "ERROR: Cannot edit binary file"

    try:
        content = p.read_text(encoding=encoding)
    except Exception as e:
        return f"ERROR reading file: {e}"

    # Exact match
    if search in content:
        new_content = content.replace(search, replace, 1)
    else:
        # Fuzzy match via existing engine
        new_content = replace_most_similar_chunk(content, search, replace)
        if new_content is None:
            similar = _find_similar_lines(search, content)
            err = f"SEARCH block failed to match in {file}"
            if similar:
                err += f"\nMost similar lines:\n```\n{similar}\n```"
            return err

    if dry_run:
        return f"[DRY RUN] Would edit {file} (search/replace)"

    try:
        p.write_text(new_content, encoding=encoding)
    except Exception as e:
        return f"ERROR writing file: {e}"

    return f"Edited {file}: search/replace applied"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registration
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    ("fs_aedit", fs_aedit, "Advanced file editing with layered matching (exact→whitespace→fuzzy)", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "File path to edit"},
            "search": {"type": "string", "description": "SEARCH block: text to find (empty to create/append)"},
            "replace": {"type": "string", "description": "REPLACE block: replacement text"},
            "dry_run": {"type": "boolean", "description": "Preview changes without writing (default: false)"},
            "fuzzy": {"type": "boolean", "description": "Enable fuzzy matching fallback (default: true)"},
            "create_if_missing": {"type": "boolean", "description": "Create file if missing and search is empty (default: true)"},
        },
        "required": ["file", "search", "replace"],
    }),
    ("fs_apply_patch", fs_apply_patch, "Apply unified diff patch to a file", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "File path to patch"},
            "patch": {"type": "string", "description": "Unified diff content (---/+++/@@ format)"},
            "dry_run": {"type": "boolean", "description": "Preview changes without writing (default: false)"},
        },
        "required": ["file", "patch"],
    }),
    ("fs_write_file", fs_write_file, "Write complete file content (whole-file rewrite)", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Complete new file content"},
            "dry_run": {"type": "boolean", "description": "Preview changes without writing (default: false)"},
        },
        "required": ["file", "content"],
    }),
    ("fs_edit_blocks", fs_edit_blocks, "Apply multiple SEARCH/REPLACE edit blocks at once", {
        "type": "object",
        "properties": {
            "blocks": {"type": "string", "description": "String with SEARCH/REPLACE blocks (filename + <<<<<<< SEARCH / ======= / >>>>>>> REPLACE)"},
            "dry_run": {"type": "boolean", "description": "Preview changes without writing (default: false)"},
            "fuzzy": {"type": "boolean", "description": "Enable fuzzy matching (default: true)"},
        },
        "required": ["blocks"],
    }),
    # New tools
    ("fs_edit_diff", fs_edit_diff, "Apply unified diff with line‑number based matching (fuzzy fallback)", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "Path to the file to edit"},
            "diff": {"type": "string", "description": "Unified diff text in ---/+++/@@ format"},
            "dry_run": {"type": "boolean", "description": "Preview without saving (default: false)"},
            "encoding": {"type": "string", "description": "File encoding (default: utf-8)"}
        },
        "required": ["file", "diff"],
    }),
    ("fs_edit", fs_edit, "Simple search‑and‑replace edit (exact→fuzzy)", {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "Path to the file to edit"},
            "search": {"type": "string", "description": "Text to find and replace"},
            "replace": {"type": "string", "description": "Replacement text"},
            "dry_run": {"type": "boolean", "description": "Preview without saving (default: false)"},
            "encoding": {"type": "string", "description": "File encoding (default: utf-8)"}
        },
        "required": ["file", "search", "replace"],
    }),
]


def register_all(registry):
    """Register all edit tools with the given registry."""
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} edit tools")

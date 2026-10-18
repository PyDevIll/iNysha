"""UTF-8-safe text I/O helpers shared by ``fs_tools`` and ``edit_tools``.

The agent works with files that may be UTF-8, UTF-8 with BOM, or legacy
cp1251 (Russian Windows).  Reads never raise on a stray non-UTF-8 byte: the
text is decoded leniently and ``meta`` reports what happened.  Writes are
atomic (temp file + ``os.replace``) so a crash cannot truncate a file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# Encoding candidates, most specific first.
_CANDIDATES = ("utf-8-sig", "utf-8", "cp1251")

# Control characters that are legitimate in text files.
_ALLOWED_CONTROL = {8, 9, 10, 12, 13}


def is_probably_binary(data: bytes, sample_size: int = 1024) -> bool:
    """Heuristic: NUL bytes or a high ratio of control characters."""
    sample = data[:sample_size]
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control = sum(1 for b in sample if b < 32 and b not in _ALLOWED_CONTROL)
    return control / len(sample) > 0.30


def _decode(data: bytes, encoding: str = "") -> tuple[str, str]:
    """Decode bytes, returning ``(text, encoding_used)``. Never raises."""
    candidates = (encoding,) if encoding else _CANDIDATES
    for enc in candidates:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace"), "utf-8/replace"


def _bad_byte(data: bytes, encoding: str) -> dict[str, Any] | None:
    """Locate the first byte that does not decode with ``encoding``."""
    try:
        data.decode(encoding)
        return None
    except UnicodeDecodeError as exc:
        return {
            "byte": data[exc.start] if exc.start < len(data) else None,
            "offset": exc.start,
        }
    except LookupError:
        return None


def _detect_encoding(data: bytes) -> str:
    """Best-effort encoding detection for repair jobs."""
    for enc in _CANDIDATES:
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    try:  # optional, much better on mixed legacy files
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        if best and best.encoding:
            return best.encoding
    except Exception:
        pass
    return "cp1251"


def read_text(
    path: str | Path,
    encoding: str = "",
    force_text: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Read a text file leniently.

    Args:
        path: File to read.
        encoding: Force a source encoding (e.g. ``cp1251``). Empty = auto.
        force_text: Decode even if the binary heuristic fires.

    Returns:
        ``(text, meta)`` where meta contains ``binary``, ``encoding``,
        ``size`` and ``bad_byte``.
    """
    p = Path(path)
    data = p.read_bytes()
    size = len(data)
    binary = is_probably_binary(data)

    if binary and not force_text:
        return "", {
            "binary": True,
            "encoding": "",
            "size": size,
            "bad_byte": {"byte": data[0] if data else None, "offset": 0},
        }

    text, used = _decode(data, encoding)
    meta: dict[str, Any] = {
        "binary": False,
        "encoding": used,
        "size": size,
        "bad_byte": _bad_byte(data, encoding) if encoding else None,
    }
    return text, meta


def write_text(
    path: str | Path,
    text: str,
    encoding: str = "utf-8",
) -> None:
    """Atomically write ``text`` (temp file in the same directory + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(p.parent), prefix=".tmp-", suffix=p.suffix or ".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def repair_encoding(path: str | Path, encoding: str = "") -> dict[str, Any]:
    """Re-encode a file as UTF-8, keeping a ``<file>.bak`` of the original bytes."""
    p = Path(path)
    data = p.read_bytes()
    source = encoding or _detect_encoding(data)
    text, used = _decode(data, source)

    backup = p.with_name(p.name + ".bak")
    backup.write_bytes(data)
    write_text(p, text, encoding="utf-8")

    return {
        "ok": True,
        "path": str(p),
        "source_encoding": used,
        "backup": str(backup),
        "original_bytes": len(data),
        "utf8_bytes": len(text.encode("utf-8")),
        "characters": len(text),
    }

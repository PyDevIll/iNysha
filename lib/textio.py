"""UTF-8-safe text I/O shared by the fs / edit / learning tools.

Windows sessions routinely mix encodings: window titles, OCR text and
hand-edited guides are Cyrillic, and a file that is *mostly* UTF-8 can still
carry a single cp1251 byte (e.g. a typographic quote). Previously one such byte
made a whole guide unreadable and un-editable ("binary file detected"), which
pushed edits through whole-file rewrites.

Policy: reads are **lenient** (never raise, fall back cp1251/replace), writes
are always **UTF-8**, and :func:`repair_encoding` rewrites a file as valid UTF-8
with a backup.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# C0/C1 control bytes — the only bytes that make a file look "binary".
# Tab (0x09), LF (0x0a) and CR (0x0d) are deliberately excluded.
_CONTROL_SET = set(range(0x00, 0x09)) | {0x0b, 0x0c} | set(range(0x0e, 0x20)) | {0x7f}

_UTF8_ALIASES = {"", "auto", "utf-8", "utf8", "utf_8", "utf-8-sig"}


def is_probably_binary(data: bytes, sample_size: int = 4096, threshold: float = 0.30) -> bool:
    """True only for genuinely binary content.

    A file that decodes as UTF-8 is text regardless of script. When it does not,
    only *control* bytes count against it — so Cyrillic (cp1251) content, whose
    bytes are all >= 0x80, is never misread as binary.
    """
    if not data:
        return False
    sample = data[:sample_size] if sample_size else data
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass
    non_printable = sum(1 for b in sample if b in _CONTROL_SET)
    return (non_printable / len(sample)) > threshold


def first_bad_byte(data: bytes, limit: int = 4096) -> dict:
    """Locate the first byte that breaks UTF-8 (for a helpful refusal message)."""
    chunk = data[:limit]
    i = 0
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError as e:
        i = e.start
        return {"offset": i, "byte": f"0x{chunk[i]:02x}", "reason": e.reason}
    return {}


def detect_encoding(data: bytes) -> dict:
    """Best-effort encoding verdict without hard external dependencies."""
    if not data:
        return {"encoding": "utf-8", "confidence": 1.0, "source": "empty"}
    for enc in ("utf-8", "utf-8-sig"):
        try:
            data.decode(enc)
            return {"encoding": enc, "confidence": 1.0, "source": "strict"}
        except UnicodeDecodeError:
            continue
    try:  # charset_normalizer ships with many HTTP stacks (requests/httpx)
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best is not None and best.encoding:
            coherence = getattr(best, "percent_coherence", None)
            conf = round(float(coherence) / 100.0, 2) if coherence else 0.5
            return {"encoding": best.encoding, "confidence": conf, "source": "charset_normalizer"}
    except Exception:
        pass
    return {"encoding": "cp1251", "confidence": 0.5, "source": "fallback-cp1251"}


def decode_text(data: bytes, encoding: str = "") -> tuple[str, str]:
    """Return ``(text, encoding_used)``; never raises on bad bytes."""
    req = (encoding or "").strip().lower()
    if req and req not in _UTF8_ALIASES:
        try:
            return data.decode(encoding, errors="replace"), encoding
        except LookupError:
            pass
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    # Mostly-UTF-8 with a stray byte: replacing the few bad bytes keeps every
    # Cyrillic character intact. Only when the data is largely non-UTF-8 do we
    # trust an encoding guess (cp1251), which would otherwise mangle valid UTF-8.
    replaced = data.decode("utf-8", errors="replace")
    bad = replaced.count("\ufffd")
    if bad / max(1, len(replaced)) < 0.05:
        return replaced, "utf-8-replace"
    verdict = detect_encoding(data)
    enc = verdict["encoding"]
    try:
        return data.decode(enc, errors="replace"), enc
    except LookupError:
        return data.decode("utf-8", errors="replace"), "utf-8-replace"


def read_text(path, encoding: str = "", force_text: bool = False) -> tuple[str, dict]:
    """Read a file leniently. Returns ``(text, meta)``.

    ``meta`` carries ``binary``/``size``/``encoding``/``utf8_ok``/``had_errors``
    and ``bad_byte`` when a strict UTF-8 decode failed. On a genuine binary file
    (and no ``force_text``) the text is ``""`` and ``meta["binary"]`` is True.
    """
    p = Path(path)
    data = p.read_bytes()
    if is_probably_binary(data) and not force_text:
        meta = {"binary": True, "size": len(data), "utf8_ok": False, "had_errors": True}
        bad = first_bad_byte(data)
        if bad:
            meta["bad_byte"] = bad
        return "", meta
    text, used = decode_text(data, encoding)
    try:
        data.decode("utf-8")
        utf8_ok = True
    except UnicodeDecodeError:
        utf8_ok = False
    meta = {"binary": False, "size": len(data), "encoding": used,
            "utf8_ok": utf8_ok, "had_errors": not utf8_ok}
    if not utf8_ok:
        bad = first_bad_byte(data)
        if bad:
            meta["bad_byte"] = bad
    return text, meta


def write_text(path, text: str) -> Path:
    """Write ``text`` as UTF-8, creating parent directories."""
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def repair_encoding(path, encoding: str = "") -> dict:
    """Rewrite a file as valid UTF-8 (with a ``.bak`` backup if it changed)."""
    p = Path(path)
    data = p.read_bytes()
    verdict = detect_encoding(data)
    target = encoding or verdict["encoding"]
    text, used = decode_text(data, target)
    clean = True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        clean = False
    bak = None
    if not clean:
        bak = p.with_name(p.name + ".bak")
        shutil.copy2(p, bak)
        p.write_text(text, encoding="utf-8")
    return {
        "path": str(p),
        "detected": verdict["encoding"],
        "confidence": verdict.get("confidence"),
        "used": used,
        "utf8_clean": clean,
        "rewritten": not clean,
        "backup": str(bak) if bak else None,
        "bytes_before": len(data),
        "bytes_after": p.stat().st_size,
    }

"""Text normalisation and Russian lemmatisation for the memory store.

SQLite's FTS5 ships with the ``unicode61`` tokeniser, which does **not**
stem: "сервером" would never match "сервер", "Андрею" would never match
"Андрей".  Every text that goes into full-text search is therefore stored
twice — as written and as a space-joined string of lemmas — and queries are
lemmatised the same way before they reach ``MATCH``.

``pymorphy3`` is imported lazily: memory search must keep working (in a
degraded, prefix-only mode) on a checkout where the dictionaries were not
installed, and the import must not slow down server startup when the memory
backend is still ``json``.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

# Letters of the Russian alphabet plus ASCII; everything else is a separator.
_WORD_RE = re.compile(r"[0-9a-zа-яё]+")
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Query strings longer than this are cut: a 200-word MATCH is never useful and
# only slows BM25 down.  Keeps one bad tool call from stalling a turn.
_MAX_QUERY_TOKENS = 24
# Shorter tokens ("в", "я", "a") carry no signal and bloat the FTS query.
_MIN_TOKEN_LEN = 2
# Prefix mode truncates lemmas so "перезапустить" and "перезапустил" meet.
DEFAULT_PREFIX_LEN = 6

_morph = None
_morph_lock = threading.Lock()
_morph_attempted = False


def has_morphology() -> bool:
    """Whether pymorphy3 dictionaries are importable."""
    return _get_morph() is not None


def _get_morph():
    """Lazily build the shared MorphAnalyzer (thread-safe, created once)."""
    global _morph, _morph_attempted
    if _morph is not None or _morph_attempted:
        return _morph
    with _morph_lock:
        if _morph is None and not _morph_attempted:
            _morph_attempted = True
            try:
                import pymorphy3

                _morph = pymorphy3.MorphAnalyzer()
            except Exception:  # pragma: no cover - depends on install
                _morph = None
    return _morph


def normalize_text(text: str) -> str:
    """Lowercase, ё→е, punctuation → spaces, single spaces."""
    if not text:
        return ""
    lowered = text.lower().replace("ё", "е")
    words = _WORD_RE.findall(lowered)
    return " ".join(words)


def normalize_name(name: str) -> str:
    """Normalise an entity name without splitting dotted/underscored tokens.

    ``"aikartinki.ru"`` must stay one token (it is a different entity from
    "aikartinki"), while ``"iНюша!"`` becomes ``"инюша"``.
    """
    if not name:
        return ""
    lowered = name.lower().replace("ё", "е")
    parts = [
        _EDGE_PUNCT_RE.sub("", part)
        for part in _WS_RE.split(lowered.strip())
    ]
    return " ".join(part for part in parts if part)


def tokenize(text: str) -> list[str]:
    """Word tokens of ``text`` after normalisation."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


@lru_cache(maxsize=50_000)
def lemma(word: str) -> str:
    """Lemma of one normalised word; the word itself if morphology is absent."""
    if not word:
        return ""
    morph = _get_morph()
    if morph is None:
        return word
    try:
        return morph.parse(word)[0].normal_form
    except Exception:  # pragma: no cover - defensive
        return word


def lemmatize(text: str) -> str:
    """Space-joined lemmas of ``text`` — the value stored in ``*_fts``."""
    return " ".join(lemma(token) for token in tokenize(text))


def fts_query(text: str, prefix_len: int | None = None) -> str:
    """Build a safe FTS5 ``MATCH`` expression from free-form user text.

    Raw user text must never reach ``MATCH``: punctuation there is FTS5
    syntax and raises ``sqlite3.OperationalError``.  Every token is quoted, so
    the only syntax in the result is ours.

    ``prefix_len`` switches to prefix matching (``"переза"*``), the fallback
    used when lemmatisation is unavailable or returns nothing.
    """
    tokens = [lemma(token) for token in tokenize(text)]
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < _MIN_TOKEN_LEN and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
        if len(unique) >= _MAX_QUERY_TOKENS:
            break
    if not unique:
        return ""
    parts: list[str] = []
    for token in unique:
        if prefix_len and len(token) > prefix_len:
            parts.append(f'"{token[:prefix_len]}"*')
        else:
            parts.append(f'"{token}"')
    return " OR ".join(parts)

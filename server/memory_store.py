"""SQLite graph for the agent's long-term memory (schema v1).

Design notes that the schema depends on (see ``data/memory_schema_draft.md``):
  * one file, no external dependency — SQLite and FTS5 ship with CPython;
  * temporal truth lives on ``edges`` (``valid_to``), not on ``nodes.status``;
  * every fact remembers where it came from (``source_episode`` + ``origin``),
    because anything written here is re-injected into the prompt forever;
  * full-text search runs on lemmas (``server.memory_text``) since
    ``unicode61`` does not stem Russian.

The store is lazy: importing this module creates nothing and reads nothing.
A connection — and the database file itself — appears on first use, so the
``json`` backend keeps the old behaviour byte for byte.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from server import config as app_config
from server.memory_text import lemmatize

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 3000

# ── DDL ────────────────────────────────────────────────────────────────
# Kept as one script: FTS triggers reference their source tables, so order
# matters and partial application would leave the index out of sync.
_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS meta(
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ep. что было сказано -------------------------------------------------
CREATE TABLE IF NOT EXISTS episodes(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT    NOT NULL,
    scope      TEXT    NOT NULL DEFAULT 'shared',
    speaker    TEXT    NOT NULL CHECK(speaker IN ('human','agent','tool','system')),
    kind       TEXT    NOT NULL CHECK(kind IN ('message','voice','tool','system','summary')),
    ts         INTEGER NOT NULL,              -- unix seconds, UTC
    text       TEXT    NOT NULL DEFAULT '',
    summary    TEXT,
    ref        TEXT,
    tone       TEXT    CHECK(tone IS NULL OR tone IN ('joke','sarcasm','serious')),
    lemma_text TEXT    NOT NULL DEFAULT '',
    hash       TEXT    NOT NULL,
    UNIQUE(chat_id, ts, speaker, hash)
);
CREATE INDEX IF NOT EXISTS episodes_chat_ts ON episodes(chat_id, ts);

-- понятия --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nodes(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT    NOT NULL CHECK(kind IN
                 ('person','project','server','rule','fact','topic','document')),
    name       TEXT    NOT NULL,
    name_norm  TEXT    NOT NULL,
    data       TEXT    NOT NULL DEFAULT '{}' CHECK(json_valid(data)),
    node_text  TEXT    NOT NULL DEFAULT '',
    lemma_text TEXT    NOT NULL DEFAULT '',
    scope      TEXT    NOT NULL DEFAULT 'shared',
    origin     TEXT    NOT NULL DEFAULT 'agent'
                 CHECK(origin IN ('operator','agent','web','tool','import')),
    confidence REAL    NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
    status     TEXT    NOT NULL DEFAULT 'active'
                 CHECK(status IN ('active','stale','disputed','retired')),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS nodes_ident ON nodes(kind, name_norm, scope);

CREATE TABLE IF NOT EXISTS node_aliases(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id    INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    alias      TEXT    NOT NULL,
    alias_norm TEXT    NOT NULL,
    UNIQUE(alias_norm, node_id)
);
CREATE INDEX IF NOT EXISTS node_aliases_node ON node_aliases(node_id);

-- связи ----------------------------------------------------------------
-- UNIQUE(src,dst,rel,valid_from) makes re-extraction idempotent: the same
-- claim learned twice from the same source does not create a duplicate row.
CREATE TABLE IF NOT EXISTS edges(
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    src            INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst            INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    rel            TEXT    NOT NULL CHECK(rel IN
                     ('likes','owns','prefers_no','depends_on',
                      'about','of','contradicts','supersedes')),
    weight         REAL    NOT NULL DEFAULT 0.5 CHECK(weight BETWEEN 0 AND 1),
    valid_from     INTEGER NOT NULL,
    valid_to       INTEGER,
    source_episode INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    origin         TEXT    NOT NULL DEFAULT 'agent'
                     CHECK(origin IN ('operator','agent','web','tool','import')),
    confidence     REAL    NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
    note           TEXT    NOT NULL DEFAULT '',
    CHECK(src <> dst),
    CHECK(valid_to IS NULL OR valid_to > valid_from),
    UNIQUE(src, dst, rel, valid_from)
);
CREATE INDEX IF NOT EXISTS edges_src ON edges(src, rel);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst, rel);
CREATE INDEX IF NOT EXISTS edges_live ON edges(dst) WHERE valid_to IS NULL;

-- сколько независимых источников подтверждают ребро ---------------------
CREATE TABLE IF NOT EXISTS edge_sources(
    edge_id    INTEGER NOT NULL REFERENCES edges(id)    ON DELETE CASCADE,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    PRIMARY KEY(edge_id, episode_id)
);

-- что уже поднимали в контекст -----------------------------------------
CREATE TABLE IF NOT EXISTS memory_index(
    item_type TEXT    NOT NULL CHECK(item_type IN ('node','edge','episode')),
    item_id   INTEGER NOT NULL,
    last_used INTEGER NOT NULL,
    hits      INTEGER NOT NULL DEFAULT 0 CHECK(hits >= 0),
    PRIMARY KEY(item_type, item_id)
);

-- кандидаты, а не факты ------------------------------------------------
CREATE TABLE IF NOT EXISTS pending_facts(
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id        TEXT,
    source_episode INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    payload        TEXT    NOT NULL CHECK(json_valid(payload)),
    reason         TEXT    NOT NULL DEFAULT '',
    status         TEXT    NOT NULL DEFAULT 'pending'
                     CHECK(status IN ('pending','accepted','rejected')),
    created_at     INTEGER NOT NULL,
    UNIQUE(chat_id, source_episode, payload)
);

-- FTS5: external content + triggers.  Without the triggers the index drifts
-- silently, which is worse than having no index at all.
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    name, node_text, lemma_text,
    content='nodes', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, name, node_text, lemma_text)
    VALUES (new.id, new.name, new.node_text, new.lemma_text);
END;
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, name, node_text, lemma_text)
    VALUES ('delete', old.id, old.name, old.node_text, old.lemma_text);
END;
CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, name, node_text, lemma_text)
    VALUES ('delete', old.id, old.name, old.node_text, old.lemma_text);
    INSERT INTO nodes_fts(rowid, name, node_text, lemma_text)
    VALUES (new.id, new.name, new.node_text, new.lemma_text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    text, summary, lemma_text,
    content='episodes', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, text, summary, lemma_text)
    VALUES (new.id, new.text, coalesce(new.summary,''), new.lemma_text);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, text, summary, lemma_text)
    VALUES ('delete', old.id, old.text, coalesce(old.summary,''), old.lemma_text);
END;
CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, text, summary, lemma_text)
    VALUES ('delete', old.id, old.text, coalesce(old.summary,''), old.lemma_text);
    INSERT INTO episodes_fts(rowid, text, summary, lemma_text)
    VALUES (new.id, new.text, coalesce(new.summary,''), new.lemma_text);
END;
"""

_TABLES = (
    "episodes",
    "nodes",
    "node_aliases",
    "edges",
    "edge_sources",
    "memory_index",
    "pending_facts",
    "nodes_fts",
    "episodes_fts",
)


def now_ts() -> int:
    """Current unix time in seconds (the only time unit used by the schema)."""
    return int(time.time())


def episode_hash(text: str) -> str:
    """Stable, short fingerprint of one utterance.

    Used by ``episodes.UNIQUE(chat_id, ts, speaker, hash)`` so that importing
    (or re-importing) the same transcript cannot duplicate rows.
    """
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


class MemoryStore:
    """One SQLite database, one connection, single writer.

    The process is single-worker (``server.runtime.worker_loop``), so a single
    connection created on the event-loop thread is enough.  ``check_same_thread``
    stays on: a stray call from another thread must fail loudly instead of
    silently corrupting the WAL, and scripts open their own ``MemoryStore``.
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._schema_version: Optional[int] = None

    # ── Lifecycle ──────────────────────────────────────────────────
    @property
    def path(self) -> Path:
        return self._path

    @property
    def exists(self) -> bool:
        return self._path.exists()

    @property
    def connection(self) -> sqlite3.Connection:
        """Open (and migrate) the database on first use."""
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    self._conn = self._open()
        return self._conn

    def _open(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        existed = self._path.exists()
        conn = sqlite3.connect(str(self._path), timeout=BUSY_TIMEOUT_MS / 1000)
        # data/ holds real conversation history; the DB must not be world-readable.
        try:
            self._path.chmod(0o600)
        except OSError as e:  # pragma: no cover - exotic filesystems
            logger.debug(f"memory: chmod 0600 failed for {self._path}: {e}")
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        # WAL keeps a crash from truncating the file; synchronous=NORMAL is the
        # usual WAL pairing (durable across process crashes, cheap per commit).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Off by default in SQLite; the edges table relies on real FKs.
        conn.execute("PRAGMA foreign_keys=ON")
        self._migrate(conn)
        logger.info(
            f"memory: store ready path={self._path} schema=v{SCHEMA_VERSION} "
            f"new_file={not existed}"
        )
        return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.execute("PRAGMA optimize")
                except Exception as e:  # pragma: no cover - best effort
                    logger.debug(f"memory: PRAGMA optimize failed: {e}")
                self._conn.close()
                self._conn = None
                logger.debug(f"memory: store closed path={self._path}")

    def __enter__(self) -> "MemoryStore":
        self.connection  # noqa: B018 - force open/migrate
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ── Schema ─────────────────────────────────────────────────────
    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = self._read_version(conn)
        if version > SCHEMA_VERSION:
            logger.error(
                f"memory: refusing to open {self._path}: database schema "
                f"v{version} > code v{SCHEMA_VERSION}"
            )
            raise RuntimeError(
                f"memory.db schema v{version} is newer than this code "
                f"(v{SCHEMA_VERSION}); upgrade the checkout instead of "
                f"downgrading the database"
            )
        if version == SCHEMA_VERSION:
            self._schema_version = version
            logger.debug(f"memory: schema already v{version}, no migration")
            return
        # v0 -> v1: create everything.  Later versions append their own step
        # here and bump SCHEMA_VERSION; never edit a shipped block in place.
        logger.info(f"memory: migrating schema v{version} -> v{SCHEMA_VERSION}")
        conn.executescript(_SCHEMA_V1)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('created_at', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (str(now_ts()),),
        )
        conn.commit()
        self._schema_version = SCHEMA_VERSION
        logger.info(
            f"memory: schema v{SCHEMA_VERSION} applied "
            f"(tables={len(_TABLES)}, fts=nodes_fts,episodes_fts)"
        )

    @staticmethod
    def _read_version(conn: sqlite3.Connection) -> int:
        has_meta = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if not has_meta:
            return 0
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        try:
            return int(row["value"]) if row else 0
        except (TypeError, ValueError):  # pragma: no cover - corrupt value
            return 0

    @property
    def schema_version(self) -> int:
        self.connection  # noqa: B018 - ensure migrated
        if self._schema_version is None:
            self._schema_version = self._read_version(self.connection)
        return self._schema_version

    # ── Episodes ───────────────────────────────────────────────────
    def add_episode(
        self,
        *,
        chat_id: str,
        speaker: str,
        kind: str,
        ts: int,
        text: str,
        summary: Optional[str] = None,
        ref: Optional[str] = None,
        tone: Optional[str] = None,
        scope: str = "shared",
        commit: bool = True,
    ) -> Optional[int]:
        """Insert one utterance; return its id, or ``None`` if already known.

        The unique key is ``(chat_id, ts, speaker, hash)``, so the same
        transcript can be imported twice, or an in-flight turn can be replayed
        after a crash, without duplicating rows.
        """
        text = (text or "").strip()
        if not text:
            return None
        lemma_text = lemmatize(text)
        if summary:
            lemma_text = f"{lemma_text} {lemmatize(summary)}".strip()
        try:
            cur = self.connection.execute(
                "INSERT OR IGNORE INTO episodes"
                "(chat_id, scope, speaker, kind, ts, text, summary, ref, tone,"
                " lemma_text, hash) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(chat_id), scope, speaker, kind, int(ts), text,
                    summary, ref, tone, lemma_text, episode_hash(text),
                ),
            )
            if commit:
                self.connection.commit()
        except sqlite3.Error as e:
            # Memory is an add-on: a broken DB must never break a turn.
            logger.warning(f"memory: episode insert failed: {e}")
            return None
        if not cur.rowcount:
            logger.debug(
                f"memory: episode already stored chat={chat_id} "
                f"ts={ts} speaker={speaker} chars={len(text)}"
            )
            return None
        logger.debug(
            f"memory: episode stored id={cur.lastrowid} chat={chat_id} "
            f"speaker={speaker} kind={kind} ts={ts} chars={len(text)}"
        )
        return int(cur.lastrowid) if cur.lastrowid else None

    # ── Diagnostics ────────────────────────────────────────────────
    def stats(self) -> dict[str, int]:
        """Row counts per table — cheap enough for a health endpoint."""
        conn = self.connection
        result: dict[str, int] = {}
        for table in _TABLES:
            row = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
            result[table] = int(row["n"])
        return result

    def integrity_ok(self) -> bool:
        row = self.connection.execute("PRAGMA integrity_check").fetchone()
        ok = bool(row) and row[0] == "ok"
        if not ok:
            logger.error(
                f"memory: integrity_check failed on {self._path}: "
                f"{row[0] if row else 'no result'}"
            )
        else:
            logger.debug(f"memory: integrity_check ok ({self._path})")
        return ok


# ── Process-wide store registry ────────────────────────────────────────
# One connection per database file. Keying by path keeps tests hermetic: a
# ContextPool pointed at a tmp dir never shares the live data/memory.db, yet
# tools and the episode log still land on the same connection.
_stores: dict[Path, MemoryStore] = {}
_store_lock = threading.Lock()


def get_store(path: Path | str | None = None) -> MemoryStore:
    """Return the shared store for ``path`` (default MEMORY_DB_PATH)."""
    resolved = Path(path or app_config.MEMORY_DB_PATH).expanduser().resolve()
    with _store_lock:
        store = _stores.get(resolved)
        if store is None:
            store = MemoryStore(resolved)
            _stores[resolved] = store
        return store


def close_store(path: Path | str | None = None) -> None:
    """Close one store, or every open store when ``path`` is omitted."""
    with _store_lock:
        if path is None:
            items = list(_stores.items())
            _stores.clear()
        else:
            resolved = Path(path).expanduser().resolve()
            store = _stores.pop(resolved, None)
            items = [(resolved, store)] if store is not None else []
    for _resolved, store in items:
        store.close()

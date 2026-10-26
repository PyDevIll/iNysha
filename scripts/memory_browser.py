"""Read-only browser for data/memory.db — no dependencies, no server restart.

Why a separate script instead of routes in the FastAPI app: the agent's web
surface is for clients and MAX, and a memory viewer needs none of that. This
one binds to 127.0.0.1, opens SQLite with ``mode=ro`` (no WAL writes, no
migrations) and reuses the same lemmatisation/FTS query builder as the agent,
so what you see here is what `memory_search` would see.

    .venv/bin/python scripts/memory_browser.py --stats
    .venv/bin/python scripts/memory_browser.py
    # → http://127.0.0.1:8765

Over SSH:
    ssh -L 8765:127.0.0.1:8765 user@host
    .venv/bin/python scripts/memory_browser.py --token secret
    # → http://127.0.0.1:8765/?token=secret

The viewer shows remembered facts (with provenance), the episode transcript,
extractor candidates (`pending_facts`), sources and table counts. Episodes
contain real conversations: treat the page as sensitive, and do not expose it
on 0.0.0.0 without a token.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import config as app_config  # noqa: E402
from server.memory_text import fts_query, preview  # noqa: E402

NODE_KINDS = ("person", "project", "server", "rule", "fact", "topic", "document")
NODE_ORIGINS = ("operator", "agent", "web", "tool", "import")
NODE_STATUSES = ("active", "stale", "disputed", "retired")
SPEAKERS = ("human", "agent", "tool", "system")
PER_PAGE = 40

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
       background: #f6f6f4; color: #1d1d1b; }
header { background: #24211e; color: #f3efe9; padding: 10px 20px; }
header a { color: #f3efe9; text-decoration: none; margin-right: 16px; font-weight: 500; }
header a:hover { text-decoration: underline; }
main { max-width: 1180px; margin: 0 auto; padding: 20px; }
h1 { font-size: 22px; margin: 0 0 14px; }
h2 { font-size: 17px; margin: 24px 0 8px; }
table { width: 100%; border-collapse: collapse; background: #fff; font-size: 14px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e6e2dc;
         vertical-align: top; }
th { background: #efece7; font-weight: 600; white-space: nowrap; }
tr:hover td { background: #faf8f5; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.muted { color: #6f6a63; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 10px;
         background: #e8e3db; font-size: 12px; white-space: nowrap; }
.badge.web { background: #dce9f7; }
.badge.tool { background: #e2efdd; }
.badge.operator { background: #f6e3c9; }
.badge.retired { background: #eee; color: #888; }
.badge.stale { background: #f7e6c8; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 8px; }
.card { background: #fff; border: 1px solid #e6e2dc; border-radius: 8px;
        padding: 10px 14px; min-width: 130px; }
.card b { display: block; font-size: 22px; }
form.filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
input[type=search], select { padding: 6px 8px; border: 1px solid #cfc8bf;
        border-radius: 6px; background: #fff; color: inherit; }
button { padding: 6px 12px; border: 1px solid #cfc8bf; border-radius: 6px;
         background: #24211e; color: #fff; cursor: pointer; }
pre { background: #fff; border: 1px solid #e6e2dc; border-radius: 8px;
      padding: 12px; overflow-x: auto; white-space: pre-wrap; }
.note { background: #fff7e6; border: 1px solid #f0d9a8; padding: 10px 12px;
        border-radius: 8px; margin: 10px 0; }
.pager { margin: 14px 0; display: flex; gap: 14px; }
"""


# ── Storage ────────────────────────────────────────────────────────────
def connect_ro(path: Path) -> sqlite3.Connection:
    """Open the database strictly read-only (no WAL writes, no migrations)."""
    resolved = Path(path).expanduser().resolve()
    conn = sqlite3.connect(
        f"file:{quote(str(resolved))}?mode=ro", uri=True, timeout=3.0
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def _count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return 0


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "episodes": _count(conn, "episodes"),
        "nodes": _count(conn, "nodes"),
        "edges": _count(conn, "edges"),
        "pending": _count(conn, "pending_facts"),
        "sources": _count(conn, "sources"),
    }


def schema_version(conn: sqlite3.Connection) -> str:
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        return str(row[0]) if row else "?"
    except sqlite3.Error:
        return "?"


def overview_data(conn: sqlite3.Connection) -> dict:
    def group(sql: str) -> list[tuple[str, int]]:
        try:
            return [(str(row[0]), int(row[1]))
                    for row in conn.execute(sql).fetchall()]
        except sqlite3.Error:
            return []

    last_ts = 0
    try:
        row = conn.execute("SELECT max(ts) FROM episodes").fetchone()
        last_ts = int(row[0] or 0)
    except (sqlite3.Error, TypeError):
        pass
    return {
        "counts": counts(conn),
        "schema": schema_version(conn),
        "last_episode": last_ts,
        "episodes_by_speaker": group(
            "SELECT speaker, count(*) FROM episodes GROUP BY 1 ORDER BY 2 DESC"
        ),
        "nodes_by_kind": group(
            "SELECT kind, count(*) FROM nodes GROUP BY 1 ORDER BY 2 DESC"
        ),
        "nodes_by_origin": group(
            "SELECT origin, count(*) FROM nodes GROUP BY 1 ORDER BY 2 DESC"
        ),
        "nodes_by_status": group(
            "SELECT status, count(*) FROM nodes GROUP BY 1 ORDER BY 2 DESC"
        ),
        "pending_by_status": group(
            "SELECT status, count(*) FROM pending_facts GROUP BY 1 ORDER BY 2 DESC"
        ),
        "sources_by_kind": group(
            "SELECT kind, count(*) FROM sources GROUP BY 1 ORDER BY 2 DESC"
        ),
    }


# ── Queries ────────────────────────────────────────────────────────────
def fetch_nodes(
    conn: sqlite3.Connection,
    q: str = "",
    kind: str = "",
    origin: str = "",
    status: str = "",
    limit: int = PER_PAGE,
    offset: int = 0,
) -> list[sqlite3.Row]:
    params: list = []
    match = fts_query(q) if q else ""
    if match:
        sql = ("SELECT n.* FROM nodes_fts JOIN nodes n "
               "ON n.id = nodes_fts.rowid WHERE nodes_fts MATCH ?")
        params.append(match)
        order = " ORDER BY bm25(nodes_fts)"
    else:
        sql = "SELECT n.* FROM nodes n WHERE 1=1"
        if q:
            sql += " AND n.name_norm LIKE ?"
            params.append(f"%{q.strip().lower()}%")
        order = " ORDER BY n.updated_at DESC"
    for column, value in (("kind", kind), ("origin", origin),
                          ("status", status)):
        if value:
            sql += f" AND n.{column} = ?"
            params.append(value)
    sql += order + " LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def fetch_node(conn: sqlite3.Connection, node_id: int) -> Optional[sqlite3.Row]:
    try:
        return conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
    except sqlite3.Error:
        return None


def fetch_node_edges(conn: sqlite3.Connection, node_id: int) -> list[sqlite3.Row]:
    try:
        return conn.execute(
            "SELECT e.*, s.name AS src_name, d.name AS dst_name "
            "FROM edges e JOIN nodes s ON s.id=e.src JOIN nodes d ON d.id=e.dst "
            "WHERE e.src=? OR e.dst=? ORDER BY e.valid_from DESC LIMIT 50",
            (node_id, node_id),
        ).fetchall()
    except sqlite3.Error:
        return []


def fetch_node_hits(conn: sqlite3.Connection, node_id: int) -> Optional[sqlite3.Row]:
    try:
        return conn.execute(
            "SELECT * FROM memory_index WHERE item_type='node' AND item_id=?",
            (node_id,),
        ).fetchone()
    except sqlite3.Error:
        return None


def fetch_episodes(
    conn: sqlite3.Connection,
    q: str = "",
    chat_id: str = "",
    speaker: str = "",
    limit: int = PER_PAGE,
    offset: int = 0,
) -> list[sqlite3.Row]:
    params: list = []
    match = fts_query(q) if q else ""
    if match:
        sql = ("SELECT e.* FROM episodes_fts JOIN episodes e "
               "ON e.id = episodes_fts.rowid WHERE episodes_fts MATCH ?")
        params.append(match)
        order = " ORDER BY bm25(episodes_fts)"
    else:
        sql = "SELECT e.* FROM episodes e WHERE 1=1"
        if q:
            sql += " AND e.text LIKE ?"
            params.append(f"%{q}%")
        order = " ORDER BY e.id DESC"
    if chat_id:
        sql += " AND e.chat_id = ?"
        params.append(chat_id)
    if speaker:
        sql += " AND e.speaker = ?"
        params.append(speaker)
    sql += order + " LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def fetch_pending(
    conn: sqlite3.Connection, status: str = "", limit: int = 200
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM pending_facts"
    params: list = []
    if status:
        sql += " WHERE status=?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def fetch_sources(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    try:
        return conn.execute(
            "SELECT * FROM sources ORDER BY last_seen DESC LIMIT ?", (int(limit),)
        ).fetchall()
    except sqlite3.Error:
        return []


# ── Rendering ──────────────────────────────────────────────────────────
def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def when(ts) -> str:
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError, OSError):
        return "—"


def badge(text: str, css: str = "") -> str:
    return f'<span class="badge {esc(css)}">{esc(text)}</span>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="muted">Пусто.</p>'
    head = "".join(f"<th>{esc(name)}</th>" for name in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def layout(title: str, body: str, schema: str = "") -> str:
    nav = (
        '<header><strong>iНюша · memory</strong> '
        '<a href="/">Обзор</a>'
        '<a href="/nodes">Факты</a>'
        '<a href="/episodes">История</a>'
        '<a href="/pending">Кандидаты</a>'
        '<a href="/sources">Источники</a>'
    )
    if schema:
        nav += f'<span class="muted">схема v{esc(schema)}</span>'
    nav += "</header>"
    return (
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>"
        f"{nav}<main>{body}</main></body></html>"
    )


def pager(prefix: str, page: int, has_next: bool) -> str:
    """``prefix`` already ends with ``?…&page=``."""
    links = []
    if page > 1:
        links.append(f'<a href="{esc(prefix)}{page - 1}">← новее</a>')
    if has_next:
        links.append(f'<a href="{esc(prefix)}{page + 1}">старее →</a>')
    return f'<div class="pager">{"".join(links)}</div>' if links else ""


def _filter_query(**kwargs) -> str:
    parts = [f"{key}={quote(str(value))}" for key, value in kwargs.items()
             if value]
    return ("&".join(parts) + "&") if parts else ""


def node_source(row: sqlite3.Row) -> tuple[str, dict]:
    try:
        payload = json.loads(row["data"] or "{}")
    except (ValueError, TypeError):
        return "", {}
    if not isinstance(payload, dict):
        return "", {}
    text = str(payload.get("text") or payload.get("note") or "")
    source = payload.get("source")
    return text, source if isinstance(source, dict) else {}


def page_overview(conn: sqlite3.Connection, db_path: Path) -> str:
    data = overview_data(conn)
    size = db_path.stat().st_size if db_path.exists() else 0
    size_text = (f"{size / 1024 / 1024:.1f} МБ" if size >= 1024 * 1024
                 else f"{size / 1024:.0f} КБ")
    c = data["counts"]
    cards = "".join(
        f'<div class="card"><b>{value}</b>{esc(label)}</div>'
        for label, value in (
            ("эпизоды", c["episodes"]), ("факты", c["nodes"]),
            ("связи", c["edges"]), ("кандидаты", c["pending"]),
            ("источники", c["sources"]),
        )
    )

    def group_rows(pairs) -> list[list[str]]:
        return [[esc(name or "—"), str(value)] for name, value in pairs]

    body = [
        "<h1>Обзор</h1>",
        f'<div class="cards">{cards}</div>',
        f'<p class="muted">{esc(db_path)} · {size_text} · '
        f'схема v{esc(data["schema"])} · последний эпизод: '
        f'{esc(when(data["last_episode"]))}</p>',
        "<h2>Эпизоды по авторам</h2>",
        table(["кто", "шт."], group_rows(data["episodes_by_speaker"])),
        "<h2>Факты</h2>",
        table(["тип", "шт."], group_rows(data["nodes_by_kind"])),
        table(["происхождение", "шт."], group_rows(data["nodes_by_origin"])),
        table(["статус", "шт."], group_rows(data["nodes_by_status"])),
        "<h2>Кандидаты экстрактора</h2>",
        table(["статус", "шт."], group_rows(data["pending_by_status"])),
        "<h2>Источники</h2>",
        table(["вид", "шт."], group_rows(data["sources_by_kind"])),
        '<p class="note">Всё открыто только на чтение. Приём кандидатов — '
        '<span class="mono">scripts/memory_pending.py</span>, изменения фактов — '
        "через агента (<span class='mono'>memory_remember</span> / "
        "<span class='mono'>memory_forget</span>).</p>",
    ]
    return layout("Память · обзор", "".join(body), data["schema"])


def page_nodes(conn: sqlite3.Connection, params: dict) -> str:
    q = (params.get("q") or [""])[0]
    kind = (params.get("kind") or [""])[0]
    origin = (params.get("origin") or [""])[0]
    status = (params.get("status") or [""])[0]
    page = max(1, int((params.get("page") or ["1"])[0] or 1))
    rows = fetch_nodes(conn, q=q, kind=kind, origin=origin, status=status,
                       limit=PER_PAGE, offset=(page - 1) * PER_PAGE)

    def options(values, current) -> str:
        out = ['<option value="">все</option>']
        out += [f'<option value="{esc(v)}"'
                f'{" selected" if v == current else ""}>{esc(v)}</option>'
                for v in values]
        return "".join(out)

    filters = (
        '<form class="filters" method="get" action="/nodes">'
        f'<input type="search" name="q" value="{esc(q)}" '
        'placeholder="поиск по фактам (FTS)">'
        f'<select name="kind">{options(NODE_KINDS, kind)}</select>'
        f'<select name="origin">{options(NODE_ORIGINS, origin)}</select>'
        f'<select name="status">{options(NODE_STATUSES, status)}</select>'
        "<button>найти</button></form>"
    )
    table_rows = []
    for row in rows:
        text, source = node_source(row)
        src = ""
        if source:
            marker = badge("web", "web") if source.get("kind") == "web" \
                else badge("file", "tool")
            src = f"{marker} {esc(preview(str(source.get('ref') or ''), 46))}"
        table_rows.append([
            f'<a class="mono" href="/node/{int(row["id"])}">{int(row["id"])}</a>',
            badge(str(row["kind"])),
            esc(row["name"]),
            esc(preview(text, 110)),
            badge(str(row["origin"]), str(row["origin"])),
            f'{float(row["confidence"]):.2f}',
            badge(str(row["status"]), str(row["status"])),
            src or '<span class="muted">—</span>',
            esc(when(row["updated_at"])),
        ])
    prefix = "/nodes?" + _filter_query(q=q, kind=kind, origin=origin,
                                       status=status) + "page="
    return layout(
        "Память · факты",
        "<h1>Факты</h1>" + filters
        + table(["id", "тип", "имя", "текст", "откуда", "conf", "статус",
                 "источник", "обновлён"], table_rows)
        + pager(prefix, page, has_next=len(rows) == PER_PAGE),
    )


def page_node(conn: sqlite3.Connection, node_id: int) -> str:
    row = fetch_node(conn, node_id)
    if row is None:
        return layout("Нет факта", "<h1>Факт не найден</h1>")
    text, source = node_source(row)
    hits = fetch_node_hits(conn, node_id)
    edges = fetch_node_edges(conn, node_id)

    info = [
        ["id", f'<span class="mono">{int(row["id"])}</span>'],
        ["тип", badge(str(row["kind"]))],
        ["имя", esc(row["name"])],
        ["происхождение", badge(str(row["origin"]), str(row["origin"]))],
        ["уверенность", f'{float(row["confidence"]):.2f}'],
        ["статус", badge(str(row["status"]), str(row["status"]))],
        ["scope", esc(row["scope"])],
        ["создан", esc(when(row["created_at"]))],
        ["обновлён", esc(when(row["updated_at"]))],
        ["в промпте", (f'{int(hits["hits"])} раз, последний — '
                       f'{esc(when(hits["last_used"]))}') if hits else "ни разу"],
    ]
    if source:
        ref = str(source.get("ref") or "")
        ref_html = (f'<a href="{esc(ref)}" rel="noreferrer">{esc(ref)}</a>'
                    if source.get("kind") == "web"
                    else f'<span class="mono">{esc(ref)}</span>')
        info.append(["источник", f'{badge(str(source.get("kind", "")))} {ref_html}'])
        if source.get("last_seen"):
            info.append(["прочитан", esc(when(source["last_seen"]))])
    try:
        raw_data = json.dumps(json.loads(row["data"] or "{}"),
                              ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        raw_data = str(row["data"])

    edge_rows = [
        [
            badge(str(edge["rel"])),
            f'<a href="/node/{int(edge["src"])}">{esc(edge["src_name"])}</a> → '
            f'<a href="/node/{int(edge["dst"])}">{esc(edge["dst_name"])}</a>',
            f'{float(edge["weight"]):.2f}',
            esc(when(edge["valid_from"])),
            esc(when(edge["valid_to"])) if edge["valid_to"] else "действует",
        ]
        for edge in edges
    ]
    body = (
        f"<h1>{esc(row['name'])}</h1>"
        + table(["поле", "значение"], [[esc(key), value] for key, value in info])
        + "<h2>Текст факта</h2>"
        + f"<pre>{esc(text or row['node_text'])}</pre>"
        + "<h2>Связи</h2>"
        + table(["связь", "узлы", "вес", "с", "по"], edge_rows)
        + "<h2>Сырой data</h2>"
        + f"<pre>{esc(raw_data)}</pre>"
    )
    return layout(f"Факт · {row['name']}", body)


def page_episodes(conn: sqlite3.Connection, params: dict) -> str:
    q = (params.get("q") or [""])[0]
    chat_id = (params.get("chat") or [""])[0]
    speaker = (params.get("speaker") or [""])[0]
    page = max(1, int((params.get("page") or ["1"])[0] or 1))
    rows = fetch_episodes(conn, q=q, chat_id=chat_id, speaker=speaker,
                          limit=PER_PAGE, offset=(page - 1) * PER_PAGE)

    options = ['<option value="">все</option>']
    options += [f'<option value="{esc(v)}"'
                f'{" selected" if v == speaker else ""}>{esc(v)}</option>'
                for v in SPEAKERS]
    filters = (
        '<form class="filters" method="get" action="/episodes">'
        f'<input type="search" name="q" value="{esc(q)}" '
        'placeholder="поиск по истории (FTS)">'
        f'<input type="search" name="chat" value="{esc(chat_id)}" placeholder="chat_id">'
        f'<select name="speaker">{"".join(options)}</select>'
        "<button>найти</button></form>"
    )
    table_rows = [
        [
            f'<span class="mono">{int(row["id"])}</span>',
            esc(when(row["ts"])),
            badge(str(row["speaker"])),
            f'<span class="mono">{esc(row["chat_id"])}</span>',
            esc(row["kind"]),
            esc(preview(row["text"], 180)),
        ]
        for row in rows
    ]
    prefix = "/episodes?" + _filter_query(q=q, chat=chat_id,
                                          speaker=speaker) + "page="
    return layout(
        "Память · история",
        "<h1>История</h1>" + filters
        + table(["id", "когда", "кто", "чат", "вид", "текст"], table_rows)
        + pager(prefix, page, has_next=len(rows) == PER_PAGE),
    )


def page_pending(conn: sqlite3.Connection, params: dict) -> str:
    status = (params.get("status") or ["pending"])[0]
    rows = fetch_pending(conn, status=status, limit=200)
    table_rows = []
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except (ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        table_rows.append([
            f'<span class="mono">{int(row["id"])}</span>',
            f'{float(payload.get("confidence") or 0):.2f}',
            badge(str(payload.get("kind") or "?")),
            esc(payload.get("name") or "—"),
            esc(preview(payload.get("text") or "", 140)),
            badge(str(row["status"]), str(row["status"])),
            esc(when(row["created_at"])),
            (f'<a href="/node/{int(row["node_id"])}">{int(row["node_id"])}</a>'
             if row["node_id"] else '<span class="muted">—</span>'),
        ])
    tabs = "".join(
        f'<a class="badge" href="/pending?status={esc(name)}">{esc(name)}</a> '
        for name in ("pending", "accepted", "rejected")
    )
    note = (
        '<p class="note">Кандидаты появляются из фонового экстрактора и никогда '
        "не попадают в промпт сами. Принять: "
        '<span class="mono">scripts/memory_pending.py --accept ID</span>.</p>'
    )
    return layout(
        "Память · кандидаты",
        "<h1>Кандидаты</h1>" + f"<p>{tabs}</p>" + note
        + table(["id", "conf", "тип", "имя", "текст", "статус", "создан", "факт"],
                table_rows),
    )


def page_sources(conn: sqlite3.Connection) -> str:
    rows = fetch_sources(conn)
    now = int(time.time())
    table_rows = []
    for row in rows:
        ref = str(row["ref"])
        ref_html = (f'<a href="{esc(ref)}" rel="noreferrer">{esc(preview(ref, 90))}</a>'
                    if row["kind"] == "web"
                    else f'<span class="mono">{esc(preview(ref, 90))}</span>')
        age = max(0, (now - int(row["last_seen"])) // 86400)
        size = row["size"]
        table_rows.append([
            badge(str(row["kind"]), str(row["kind"])),
            ref_html,
            esc(when(row["last_seen"])),
            f"{age} дн.",
            str(int(row["reads"])),
            f"{size / 1024:.0f} КБ" if size else "—",
        ])
    note = (
        '<p class="note">Факт может ссылаться только на источник, прочитанный '
        f"за последние {app_config.MEMORY_SOURCE_WINDOW_DAYS} дней. Старые "
        "строки остаются здесь для истории.</p>"
    )
    return layout(
        "Память · источники",
        "<h1>Источники</h1>" + note
        + table(["вид", "ссылка", "последнее чтение", "возраст", "чтений", "размер"],
                table_rows),
    )


# ── HTTP ───────────────────────────────────────────────────────────────
class MemoryHandler(BaseHTTPRequestHandler):
    server_version = "iNyshaMemory/1.0"

    def _cookie_token(self) -> str:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get("memtoken")
        return morsel.value if morsel else ""

    def _send(self, status: int, body: str, cookie: str = "") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query)
        token = getattr(self.server, "token", "")
        cookie = ""
        if token:
            supplied = (params.get("token") or [""])[0] or self._cookie_token()
            if supplied != token:
                self._send(401, layout(
                    "Нужен токен",
                    "<h1>401</h1><p>Открой адрес с "
                    "<span class='mono'>?token=…</span>.</p>",
                ))
                return
            if params.get("token"):
                cookie = f"memtoken={token}; Path=/; HttpOnly; SameSite=Strict"

        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        db_path: Path = self.server.db_path  # type: ignore[attr-defined]
        try:
            conn = connect_ro(db_path)
        except sqlite3.Error as e:
            self._send(500, layout(
                "Ошибка", f"<h1>Не открыть базу</h1><pre>{esc(e)}</pre>"))
            return
        try:
            if parsed.path == "/":
                status, body = 200, page_overview(conn, db_path)
            elif parsed.path == "/nodes":
                status, body = 200, page_nodes(conn, params)
            elif parsed.path.startswith("/node/"):
                try:
                    node_id = int(parsed.path.rsplit("/", 1)[-1])
                except ValueError:
                    status, body = 404, layout(
                        "404", "<h1>404</h1><p>Нет такого факта.</p>")
                else:
                    status, body = 200, page_node(conn, node_id)
            elif parsed.path == "/episodes":
                status, body = 200, page_episodes(conn, params)
            elif parsed.path == "/pending":
                status, body = 200, page_pending(conn, params)
            elif parsed.path == "/sources":
                status, body = 200, page_sources(conn)
            else:
                status, body = 404, layout(
                    "404", "<h1>404</h1><p>Нет такой страницы.</p>")
        except Exception as e:  # страница не должна ронять сервер
            status, body = 500, layout(
                "Ошибка", f"<h1>Ошибка страницы</h1><pre>{esc(e)}</pre>")
        finally:
            conn.close()
        self._send(status, body, cookie)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        self._send(405, layout("Только чтение",
                               "<h1>405</h1><p>Просмотрщик только читает.</p>"))

    def log_message(self, fmt: str, *args) -> None:
        # Path only: the query string may contain the access token.
        path = urlsplit(self.path).path
        sys.stderr.write(f"{self.address_string()} {self.command} {path}\n")


def serve(db_path: Path, host: str, port: int, token: str = "") -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), MemoryHandler)
    httpd.db_path = db_path  # type: ignore[attr-defined]
    httpd.token = token  # type: ignore[attr-defined]
    return httpd


def print_stats(conn: sqlite3.Connection, db_path: Path) -> None:
    data = overview_data(conn)
    size = db_path.stat().st_size if db_path.exists() else 0
    print(f"база: {db_path} ({size / 1024 / 1024:.1f} МБ, схема v{data['schema']})")
    print(f"последний эпизод: {when(data['last_episode'])}")
    for label, key in (("эпизоды", "episodes"), ("факты", "nodes"),
                       ("связи", "edges"), ("кандидаты", "pending"),
                       ("источники", "sources")):
        print(f"  {label:<10} {data['counts'][key]}")
    for title, key in (("эпизоды по авторам", "episodes_by_speaker"),
                       ("факты по типу", "nodes_by_kind"),
                       ("факты по происхождению", "nodes_by_origin"),
                       ("кандидаты по статусу", "pending_by_status"),
                       ("источники по виду", "sources_by_kind")):
        if data[key]:
            pairs = ", ".join(f"{name}={value}" for name, value in data[key])
            print(f"  {title}: {pairs}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(app_config.MEMORY_DB_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.getenv("MEMORY_BROWSER_TOKEN", ""))
    parser.add_argument("--stats", action="store_true",
                        help="напечатать содержимое и выйти")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        print(f"нет базы: {db_path}")
        return 1
    if args.stats:
        conn = connect_ro(db_path)
        try:
            print_stats(conn, db_path)
        finally:
            conn.close()
        return 0
    if args.host not in ("127.0.0.1", "::1", "localhost") and not args.token:
        print("Внимание: база открыта не только локально и без токена. "
              "Добавь --token или --host 127.0.0.1.")

    httpd = serve(db_path, args.host, args.port, args.token)
    url = f"http://{args.host}:{args.port}/"
    if args.token:
        url += f"?token={args.token}"
    print(f"Память: {db_path}")
    print(f"Открыто: {url}  (Ctrl+C — остановить)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

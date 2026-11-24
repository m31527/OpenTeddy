"""
Knowledge base: the owner's own reference material (FAQs, product notes,
policies) that answers are taken from. Optional — Settings → 知識庫, off
by default.

  datasets   named collections, managed in the web UI's 知識庫 tab or the
             /knowledge API; each holds the rows of the files uploaded to it
  upload     .csv / .tsv (a question/answer column pair becomes one entry
             per row; any other table: one entry per row, "column: value")
             and .txt / .md (paragraphs, cut to ~600 characters). The same
             file name uploaded again replaces its rows.
  search     keyword search that works on Chinese without a tokenizer
             (character pairs, BM25), plus meaning search when an
             embedding model is set (knowledge_embed_model, through
             Ollama's /api/embed); the two rankings are merged (RRF)
  scope      each session picks its datasets (the chat box's ＋ → 知識庫,
             PUT /sessions/{id}/knowledge); a dataset marked "always"
             (off unless the owner ticks it) is added to every session;
             a Task API call that names `datasets: [...]` uses exactly
             those for that one task ([] = none)
  answers    the entries that match the person's words go into the chat
             lane's, the planner's and the executor's prompt, with the
             rule to answer from them and not to fill in what they don't
             say; the agent can search further with knowledge_search

A dataset is shared by everyone who uses this OpenTeddy, like its skills,
so it is no place for personal data (privacy.py keeps that per person).
"""
from __future__ import annotations

import contextvars
import csv
import io
import json
import logging
import math
import re
import secrets
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ITEMS_PER_DATASET = 20000
MAX_CELL_CHARS = 4000
CHUNK_CHARS = 600                    # a text file is cut into pieces about this long
BLOCK_CHARS = 2600                   # the most a prompt gets from the knowledge base
ENTRY_CHARS = 700                    # one entry's text in a prompt
EXTENSIONS = (".csv", ".tsv", ".txt", ".md")
TOOL_NAMES = frozenset({"knowledge_search"})     # hidden from a task without datasets

_TABLES = (
    "CREATE TABLE IF NOT EXISTS knowledge_datasets ("
    "  id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',"
    "  always INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS knowledge_items ("
    "  id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL, source TEXT NOT NULL,"
    "  ref TEXT NOT NULL DEFAULT '', title TEXT NOT NULL DEFAULT '', body TEXT NOT NULL,"
    "  embedding BLOB, embed_model TEXT NOT NULL DEFAULT '', added_at TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_items_ds ON knowledge_items(dataset_id, source)",
    # Which datasets each session answers from — chosen per session (the chat
    # box's ＋ → 知識庫, or PUT /sessions/{id}/knowledge).
    "CREATE TABLE IF NOT EXISTS knowledge_session_datasets ("
    "  session_id TEXT NOT NULL, dataset_id TEXT NOT NULL, added_at TEXT NOT NULL,"
    "  PRIMARY KEY (session_id, dataset_id))",
)

_tracker: Any = None
_scope: contextvars.ContextVar[Optional[Tuple[str, ...]]] = contextvars.ContextVar("knowledge_scope", default=None)


class KnowledgeError(Exception):
    """Something to tell the person as it is."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enabled() -> bool:
    from config import config
    return bool(getattr(config, "knowledge_enabled", False))


def embed_model() -> str:
    from config import config
    return (getattr(config, "knowledge_embed_model", "") or "").strip()


# ── Storage ──────────────────────────────────────────────────────────────────

async def _db():
    if _tracker is None:
        raise RuntimeError("knowledge not started")
    return _tracker.db


async def start(tracker: Any) -> None:
    global _tracker
    _tracker = tracker
    for sql in _TABLES:
        await tracker.db.execute(sql)
    await tracker.db.commit()


def _row(cur_desc: Any, row: Sequence[Any]) -> Dict[str, Any]:
    return {d[0]: v for d, v in zip(cur_desc, row)}


async def list_datasets() -> List[Dict[str, Any]]:
    db = await _db()
    out: List[Dict[str, Any]] = []
    async with db.execute("SELECT id, name, description, always, created_at, updated_at "
                          "FROM knowledge_datasets ORDER BY created_at") as cur:
        rows = [_row(cur.description, r) for r in await cur.fetchall()]
    for d in rows:
        d["always"] = bool(d["always"])
        async with db.execute(
                "SELECT source, COUNT(*), SUM(embedding IS NOT NULL), MAX(added_at) FROM knowledge_items "
                "WHERE dataset_id=? GROUP BY source ORDER BY MIN(id)", (d["id"],)) as cur:
            srcs = await cur.fetchall()
        d["sources"] = [{"name": s, "items": n, "semantic": int(e or 0), "uploaded_at": at}
                        for s, n, e, at in srcs]
        d["items"] = sum(s["items"] for s in d["sources"])
        d["semantic"] = sum(s["semantic"] for s in d["sources"])
        out.append(d)
    return out


async def get_dataset(ref: str) -> Optional[Dict[str, Any]]:
    """By id, or by its exact name (what an API caller is likely to write)."""
    ref = (ref or "").strip()
    if not ref:
        return None
    all_ds = await list_datasets()
    return (next((d for d in all_ds if d["id"] == ref), None)
            or next((d for d in all_ds if d["name"].strip().lower() == ref.lower()), None))


async def resolve(refs: Iterable[str]) -> List[str]:
    """Dataset ids for the names or ids given; KnowledgeError names the unknown ones."""
    ids, unknown = [], []
    for r in refs:
        d = await get_dataset(str(r))
        if d:
            if d["id"] not in ids:
                ids.append(d["id"])
        else:
            unknown.append(str(r))
    if unknown:
        raise KnowledgeError(f"找不到這些資料集：{', '.join(unknown)}（GET /knowledge 列出全部）")
    return ids


async def create_dataset(name: str, description: str = "", always: bool = False) -> Dict[str, Any]:
    name = (name or "").strip()[:80]
    if not name:
        raise KnowledgeError("資料集要有名稱。")
    if await get_dataset(name):
        raise KnowledgeError(f"已經有叫「{name}」的資料集了。")
    db = await _db()
    ds_id = "kb-" + secrets.token_hex(4)
    now = _now()
    await db.execute("INSERT INTO knowledge_datasets(id, name, description, always, created_at, updated_at) "
                     "VALUES (?,?,?,?,?,?)", (ds_id, name, (description or "").strip()[:500], int(bool(always)), now, now))
    await db.commit()
    return await get_dataset(ds_id)  # type: ignore[return-value]


async def update_dataset(ds_id: str, name: Optional[str] = None, description: Optional[str] = None,
                         always: Optional[bool] = None) -> Dict[str, Any]:
    d = await get_dataset(ds_id)
    if not d or d["id"] != ds_id:
        raise KnowledgeError("找不到這個資料集。")
    if name is not None:
        name = name.strip()[:80]
        if not name:
            raise KnowledgeError("資料集要有名稱。")
        other = await get_dataset(name)
        if other and other["id"] != ds_id:
            raise KnowledgeError(f"已經有叫「{name}」的資料集了。")
    db = await _db()
    await db.execute(
        "UPDATE knowledge_datasets SET name=COALESCE(?, name), description=COALESCE(?, description), "
        "always=COALESCE(?, always), updated_at=? WHERE id=?",
        (name, None if description is None else description.strip()[:500],
         None if always is None else int(bool(always)), _now(), ds_id))
    await db.commit()
    _changed(ds_id)
    return await get_dataset(ds_id)  # type: ignore[return-value]


async def delete_dataset(ds_id: str) -> bool:
    db = await _db()
    await db.execute("DELETE FROM knowledge_items WHERE dataset_id=?", (ds_id,))
    await db.execute("DELETE FROM knowledge_session_datasets WHERE dataset_id=?", (ds_id,))
    cur = await db.execute("DELETE FROM knowledge_datasets WHERE id=?", (ds_id,))
    await db.commit()
    _changed(ds_id)
    return bool(cur.rowcount)


async def delete_source(ds_id: str, source: str) -> int:
    db = await _db()
    cur = await db.execute("DELETE FROM knowledge_items WHERE dataset_id=? AND source=?", (ds_id, source))
    await db.commit()
    _changed(ds_id)
    return cur.rowcount or 0


async def delete_item(item_id: int) -> bool:
    db = await _db()
    async with db.execute("SELECT dataset_id FROM knowledge_items WHERE id=?", (int(item_id),)) as cur:
        row = await cur.fetchone()
    if not row:
        return False
    await db.execute("DELETE FROM knowledge_items WHERE id=?", (int(item_id),))
    await db.commit()
    _changed(row[0])
    return True


async def list_items(ds_id: str, offset: int = 0, limit: int = 50, q: str = "") -> Dict[str, Any]:
    db = await _db()
    where, args = "dataset_id=?", [ds_id]
    if q.strip():
        where += " AND (title LIKE ? OR body LIKE ?)"
        args += [f"%{q.strip()}%"] * 2
    async with db.execute(f"SELECT COUNT(*) FROM knowledge_items WHERE {where}", args) as cur:
        total = (await cur.fetchone())[0]
    async with db.execute(
            f"SELECT id, source, ref, title, body, embedding IS NOT NULL FROM knowledge_items WHERE {where} "
            "ORDER BY id LIMIT ? OFFSET ?", args + [max(1, min(int(limit), 200)), max(0, int(offset))]) as cur:
        rows = await cur.fetchall()
    return {"total": total, "items": [{"id": i, "source": s, "ref": r, "title": t, "body": b, "semantic": bool(e)}
                                      for i, s, r, t, b, e in rows]}


# ── Reading files ────────────────────────────────────────────────────────────

def decode(data: bytes) -> str:
    """UTF-8 (with or without BOM), else Big5 (Excel on a Traditional
    Chinese Windows saves CSV that way), else GB18030."""
    for enc in ("utf-8-sig", "cp950", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise KnowledgeError("讀不懂這個檔案的文字編碼；請存成 UTF-8 再上傳。")


_Q_NAMES = {"question", "questions", "q", "問題", "問", "題目", "提問", "常見問題", "faq", "標題", "title", "主題", "關鍵字"}
_A_NAMES = {"answer", "answers", "a", "答案", "答", "回答", "解答", "回覆", "說明", "內容", "content", "reply", "response"}


# A column whose name contains one of these words, when none is named exactly
# (「客戶常見問題」 / 「回答內容」 in the first real FAQ, 11/20); a category or id
# column (「問題分類」, 「主題分類」, 「編號」) is never the question.
_Q_PARTS = ("問題", "提問", "題目", "question", "faq")
_A_PARTS = ("回答", "答案", "解答", "回覆", "答覆", "answer", "reply")
_A_WEAK = ("說明", "內容", "content", "description")
_NOT_TEXT = re.compile(r"(分類|類別|類型|編號|代號|序號|category|type|\bid\b|no\.?$)", re.I)


def _col(headers: List[str], names: set, parts: Sequence[str] = (), weak: Sequence[str] = (),
         skip: int = -1) -> int:
    norm = [re.sub(r"[\s_\-:：()（）]", "", h).lower() for h in headers]
    for i, h in enumerate(norm):
        if h in names and i != skip:
            return i
    for words in (parts, weak):
        for i, h in enumerate(norm):
            if (i != skip and words and any(w in h for w in words)
                    and not _NOT_TEXT.search(headers[i].strip())):
                return i
    return -1


def parse_csv(text: str, delimiter: Optional[str] = None) -> List[Dict[str, str]]:
    sample = text[:20000]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except csv.Error:
            delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter)]
    rows = [[c.strip() for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows:
        raise KnowledgeError("這個 CSV 是空的。")
    headers = [h or f"欄{i + 1}" for i, h in enumerate(rows[0])]
    body_rows = rows[1:]
    if not body_rows:
        raise KnowledgeError("這個 CSV 只有標題列，沒有內容。")
    qi = _col(headers, _Q_NAMES, _Q_PARTS)
    ai = _col(headers, _A_NAMES, _A_PARTS, _A_WEAK, skip=qi)
    if qi == ai:
        ai = -1
    out: List[Dict[str, str]] = []
    for n, r in enumerate(body_rows, start=2):
        cells = (r + [""] * len(headers))[:max(len(headers), len(r))]
        cells = [c[:MAX_CELL_CHARS] for c in cells]
        hdr = headers + [f"欄{i + 1}" for i in range(len(headers), len(cells))]
        if qi >= 0 and ai >= 0:
            title, answer = cells[qi], cells[ai]
            if not (title or answer):
                continue
            extra = [f"{hdr[i]}：{c}" for i, c in enumerate(cells) if c and i not in (qi, ai)]
            body = "\n".join(([answer] if answer else []) + extra)
        else:
            filled = [(hdr[i], c) for i, c in enumerate(cells) if c]
            if not filled:
                continue
            title = filled[0][1][:200]
            body = "\n".join(f"{h}：{c}" for h, c in filled)
        out.append({"ref": f"第 {n} 列", "title": title[:500], "body": body or title})
    return out


_SENT_END = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+|\n")
_CJK = re.compile(r"[㐀-鿿]")


def parse_text(text: str) -> List[Dict[str, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        raise KnowledgeError("這個檔案沒有文字。")
    pieces: List[str] = []
    for p in paras:
        if len(p) <= CHUNK_CHARS * 1.6:
            pieces.append(p)
            continue
        cur = ""
        for s in (x.strip() for x in _SENT_END.split(p)):
            if not s:
                continue
            if cur and len(cur) + len(s) > CHUNK_CHARS:
                pieces.append(cur)
                cur = ""
            cjk = _CJK.match(s) or (cur and _CJK.match(cur[-1]))
            cur += s if not cur else ("" if cjk else " ") + s
            while len(cur) > CHUNK_CHARS * 1.6:                # one endless sentence
                pieces.append(cur[:CHUNK_CHARS])
                cur = cur[CHUNK_CHARS:]
        if cur.strip():
            pieces.append(cur.strip())
    # A heading line goes with the paragraph under it; short neighbours belong together.
    def _heading_like(line: str) -> bool:
        line = line.strip()
        return 0 < len(line) <= 40 and not re.search(r"[。！？.!?；;：:，,]$", line)

    headed: List[Tuple[str, bool]] = []               # (text, starts with a heading)
    heading = ""
    for p in pieces:
        if "\n" not in p and _heading_like(p):
            heading += p + "\n"
            continue
        own = "\n" in p and _heading_like(p.split("\n", 1)[0])   # 「聯絡方式\n電話 …」
        headed.append((heading + p, bool(heading) or own))
        heading = ""
    if heading.strip():
        headed.append((heading.strip(), True))
    merged: List[str] = []
    for p, starts_section in headed:
        if (merged and not starts_section and len(merged[-1]) + len(p) + 1 <= CHUNK_CHARS
                and (len(merged[-1]) < 120 or len(p) < 120)):
            merged[-1] += "\n" + p
        else:
            merged.append(p)
    out = []
    for n, p in enumerate(merged, start=1):
        first = p.split("\n", 1)[0].strip()
        title = first if len(first) <= 60 and "\n" in p else ""
        out.append({"ref": f"第 {n} 段", "title": title, "body": p})
    return out


def parse_file(filename: str, data: bytes) -> List[Dict[str, str]]:
    name = (filename or "").strip()
    low = name.lower()
    if not low.endswith(EXTENSIONS):
        raise KnowledgeError("只能上傳 .csv、.tsv、.txt 或 .md 檔。")
    if len(data) > MAX_FILE_BYTES:
        raise KnowledgeError(f"檔案太大（上限 {MAX_FILE_BYTES // (1024 * 1024)} MB）。")
    text = decode(data)
    if low.endswith(".csv"):
        return parse_csv(text)
    if low.endswith(".tsv"):
        return parse_csv(text, "\t")
    return parse_text(text)


async def add_file(ds_id: str, filename: str, data: bytes) -> Dict[str, Any]:
    """Read the file into the dataset; the same file name replaces its rows."""
    d = await get_dataset(ds_id)
    if not d or d["id"] != ds_id:
        raise KnowledgeError("找不到這個資料集。")
    source = re.sub(r"[\\/]+", "_", (filename or "").strip())[-120:] or "upload.txt"
    entries = parse_file(source, data)
    db = await _db()
    async with db.execute("SELECT COUNT(*) FROM knowledge_items WHERE dataset_id=? AND source=?",
                          (ds_id, source)) as cur:
        old = (await cur.fetchone())[0]
    if d["items"] - old + len(entries) > MAX_ITEMS_PER_DATASET:
        raise KnowledgeError(f"一個資料集最多 {MAX_ITEMS_PER_DATASET} 筆；這個檔案有 {len(entries)} 筆。")
    vectors, note = await _embed_entries(entries)
    model = embed_model() if vectors else ""
    await db.execute("DELETE FROM knowledge_items WHERE dataset_id=? AND source=?", (ds_id, source))
    now = _now()
    await db.executemany(
        "INSERT INTO knowledge_items(dataset_id, source, ref, title, body, embedding, embed_model, added_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [(ds_id, source, e["ref"], e["title"], e["body"], _pack(vectors[i]) if vectors else None, model, now)
         for i, e in enumerate(entries)])
    await db.execute("UPDATE knowledge_datasets SET updated_at=? WHERE id=?", (now, ds_id))
    await db.commit()
    _changed(ds_id)
    logger.info("Knowledge: %s ← %s (%d entries, replaced %d, semantic=%s)", ds_id, source, len(entries), old,
                bool(vectors))
    return {"source": source, "added": len(entries), "replaced": old, "semantic": bool(vectors), "note": note,
            "sample": entries[:3]}


async def reindex(ds_id: str) -> Dict[str, Any]:
    """Meaning-search vectors for entries made without one (or with another model)."""
    model = embed_model()
    if not model:
        raise KnowledgeError("還沒有設定語意搜尋模型（設定 → 知識庫語意模型）。")
    db = await _db()
    async with db.execute("SELECT id, title, body FROM knowledge_items WHERE dataset_id=? "
                          "AND (embedding IS NULL OR embed_model != ?)", (ds_id, model)) as cur:
        rows = await cur.fetchall()
    done = 0
    for i in range(0, len(rows), 64):
        part = rows[i:i + 64]
        vecs = await embed([_embed_text(t, b) for _, t, b in part])
        await db.executemany("UPDATE knowledge_items SET embedding=?, embed_model=? WHERE id=?",
                             [(_pack(v), model, rid) for (rid, _, _), v in zip(part, vecs)])
        done += len(part)
    await db.commit()
    _changed(ds_id)
    return {"updated": done, "model": model}


# ── Meaning search (optional) ────────────────────────────────────────────────

def _pack(vec: Sequence[float]) -> bytes:
    import numpy as np
    return np.asarray(vec, dtype=np.float32).tobytes()


def _embed_text(title: str, body: str) -> str:
    return ((title + "\n") if title else "") + body[:1500]


async def embed(texts: List[str]) -> List[List[float]]:
    """Unit vectors from Ollama's /api/embed for the configured model."""
    import httpx
    import numpy as np
    from config import config
    model = embed_model()
    if not model:
        raise KnowledgeError("還沒有設定語意搜尋模型。")
    url = f"{config.gemma_base_url.rstrip('/')}/api/embed"
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), 32):
            r = await client.post(url, json={"model": model, "input": texts[i:i + 32],
                                             "keep_alive": getattr(config, "ollama_keep_alive", "24h")})
            if r.status_code == 404 or "not found" in r.text[:300].lower():
                raise KnowledgeError(f"Ollama 上沒有「{model}」這個模型（ollama pull {model}）。")
            r.raise_for_status()
            for v in r.json().get("embeddings") or []:
                a = np.asarray(v, dtype=np.float32)
                n = float(np.linalg.norm(a)) or 1.0
                out.append((a / n).tolist())
    if len(out) != len(texts):
        raise KnowledgeError("語意模型回傳的數量不對。")
    return out


async def _embed_entries(entries: List[Dict[str, str]]) -> Tuple[Optional[List[List[float]]], str]:
    if not embed_model():
        return None, ""
    try:
        return await embed([_embed_text(e["title"], e["body"]) for e in entries]), ""
    except Exception as exc:  # noqa: BLE001 — keyword search still works
        logger.warning("Knowledge: embedding failed, keyword search only: %s", exc)
        return None, f"語意索引沒有建立（{exc}），先用關鍵字搜尋；之後可以按「重建語意索引」。"


# ── Keyword search ───────────────────────────────────────────────────────────

# Question words and fillers: in a small FAQ they'd match every row.
_STOP = {"請問", "問一", "一下", "什麼", "甚麼", "怎麼", "怎樣", "如何", "可以", "是否", "有沒", "沒有", "我想", "想要",
         "知道", "為什", "麼時", "時候", "多少", "多久", "哪裡", "哪些", "哪個", "這個", "那個", "我們", "你們", "是不",
         "不是", "能不", "不能", "要怎", "樣才", "的話", "請教", "為何", "麼辦", "有哪", "有什", "哪家", "the", "is", "are", "what", "how", "can", "do", "does", "to", "of",
         "and", "or", "in", "on", "for", "my", "your", "a", "an", "i", "you", "it", "be"}
_TOKEN = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*|[㐀-䶿一-鿿豈-﫿]+")


def tokens(text: str) -> List[str]:
    t = unicodedata.normalize("NFKC", text or "").lower().replace("臺", "台")
    out: List[str] = []
    for m in _TOKEN.finditer(t):
        s = m.group(0)
        if s.isascii():
            if len(s) >= 2 or s.isdigit():
                out.append(s)
        elif len(s) == 1:
            out.append(s)
        else:
            out.extend(s[i:i + 2] for i in range(len(s) - 1))
    return [x for x in out if x not in _STOP]


@dataclass
class _Doc:
    id: int
    dataset_id: str
    source: str
    ref: str
    title: str
    body: str
    tf: Dict[str, int]
    length: int
    vec: Any = None
    model: str = ""


@dataclass
class _Index:
    docs: List[_Doc] = field(default_factory=list)
    df: Dict[str, int] = field(default_factory=dict)
    avg_len: float = 1.0
    built: float = 0.0


_indexes: Dict[str, _Index] = {}


def _changed(ds_id: str) -> None:
    _indexes.pop(ds_id, None)


async def _index(ds_id: str) -> _Index:
    hit = _indexes.get(ds_id)
    if hit is not None:
        return hit
    import numpy as np
    db = await _db()
    async with db.execute("SELECT id, source, ref, title, body, embedding, embed_model FROM knowledge_items "
                          "WHERE dataset_id=? ORDER BY id", (ds_id,)) as cur:
        rows = await cur.fetchall()
    idx = _Index(built=time.time())
    for rid, source, ref, title, body, emb, model in rows:
        toks = tokens(title) * 2 + tokens(body)              # a question's own words count double
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            idx.df[t] = idx.df.get(t, 0) + 1
        idx.docs.append(_Doc(rid, ds_id, source, ref, title, body, tf, len(toks) or 1,
                             np.frombuffer(emb, dtype=np.float32) if emb else None, model or ""))
    idx.avg_len = (sum(d.length for d in idx.docs) / len(idx.docs)) if idx.docs else 1.0
    _indexes[ds_id] = idx
    return idx


@dataclass
class Hit:
    id: int
    dataset_id: str
    dataset: str
    source: str
    ref: str
    title: str
    body: str
    score: float
    coverage: float
    similarity: float

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "dataset_id": self.dataset_id, "dataset": self.dataset, "source": self.source,
                "ref": self.ref, "title": self.title, "body": self.body, "score": round(self.score, 4),
                "keyword_match": round(self.coverage, 3), "meaning_match": round(self.similarity, 3)}


COVERAGE_MIN = 0.25          # share of the question's (weighted) words an entry must contain
# Cosine for a meaning match to count on its own, calibrated on bge-m3 (home,
# 11/20, an 8-row FAQ): 11 paraphrases scored 0.535–0.814 on their entry
# (退錢 ↔ 退款 0.81, 能刷卡嗎 ↔ 付款方式 0.65), 10 unrelated questions 0.33–0.52
# (「你好」 0.52). Another embedding model would need its own numbers.
SIMILARITY_MIN = 0.58
SIMILARITY_LOOSE = 0.50      # what the agent's search tool also gets


async def search(query: str, dataset_ids: Sequence[str], k: int = 5, loose: bool = False) -> List[Hit]:
    """Best entries for the query across the datasets. `loose` (the tool)
    also returns weaker matches; prompts get only the strong ones."""
    ids = [d for d in dataset_ids if d]
    q = tokens(query)
    if not ids or not (q or (query or "").strip()):
        return []
    names = {d["id"]: d["name"] for d in await list_datasets()}
    docs: List[_Doc] = []
    df: Dict[str, int] = {}
    total_len = 0.0
    for ds in ids:
        idx = await _index(ds)
        docs += idx.docs
        for t, n in idx.df.items():
            df[t] = df.get(t, 0) + n
        total_len += idx.avg_len * len(idx.docs)
    if not docs:
        return []
    n_docs = len(docs)
    avg = total_len / n_docs if n_docs else 1.0
    def idf(t: str) -> float:
        n = max(df.get(t, 0), 1)
        return math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
    # A pair no entry has is mostly one that straddles two words (「要開」 in
    # 「我要開發票」) and can never match: it counts for little, so a chatty
    # question isn't sunk by its own wording.
    top_idf = idf("\0")
    k1, b = 1.2, 0.75
    kw: Dict[int, Tuple[float, float]] = {}
    strong_kw: set = set()
    # The whole question, and each clause of it on its own: in 「顧客問：退貨的
    # 運費是誰出？請回覆他。」 the request around the question would otherwise
    # water the match down below the bar.
    clauses = [c for c in re.split(r"[，,。．.！!？?；;：:\n]+", query or "") if len(tokens(c)) >= 2]
    for variant in [q] + ([tokens(c) for c in clauses] if len(clauses) > 1 else []):
        qset = list(dict.fromkeys(variant))
        q_weight = sum(idf(t) if df.get(t) else 0.3 * top_idf for t in qset) or 1.0
        for i, d in enumerate(docs):
            s, cov, matched, rare = 0.0, 0.0, 0, False
            for t in qset:
                f = d.tf.get(t)
                if not f:
                    continue
                w = idf(t)
                cov += w
                matched += 1
                rare = rare or df.get(t, 0) == 1
                s += w * f * (k1 + 1) / (f + k1 * (1 - b + b * d.length / avg))
            if s > 0:
                old = kw.get(i, (0.0, 0.0))
                kw[i] = (max(old[0], s), max(old[1], cov / q_weight))
                # One shared word that several entries have (「商品」 in 「商品推薦有哪些」)
                # isn't a match; two words, or one only this entry has, is.
                if matched >= 2 or rare:
                    strong_kw.add(i)

    sim: Dict[int, float] = {}
    model = embed_model()
    if model and any(d.vec is not None and d.model == model for d in docs):
        try:
            import numpy as np
            qv = np.asarray((await embed([query]))[0], dtype=np.float32)
            for i, d in enumerate(docs):
                if d.vec is not None and d.model == model and d.vec.shape == qv.shape:
                    sim[i] = float(np.dot(d.vec, qv))
        except Exception as exc:  # noqa: BLE001
            logger.info("Knowledge: meaning search skipped: %s", exc)

    cov_min = 0.18 if loose else COVERAGE_MIN
    sim_min = SIMILARITY_LOOSE if loose else SIMILARITY_MIN
    keep = ({i for i, (_, c) in kw.items() if c >= cov_min and (loose or i in strong_kw)}
            | {i for i, s in sim.items() if s >= sim_min})
    if not keep:
        return []
    # Reciprocal-rank fusion of the two rankings.
    fused: Dict[int, float] = {i: 0.0 for i in keep}
    for ranking in (sorted(kw, key=lambda i: -kw[i][0]), sorted(sim, key=lambda i: -sim[i])):
        for r, i in enumerate(ranking):
            if i in fused:
                fused[i] += 1.0 / (60 + r)
    best = sorted(fused, key=lambda i: -fused[i])[:max(1, min(int(k or 5), 20))]
    return [Hit(docs[i].id, docs[i].dataset_id, names.get(docs[i].dataset_id, docs[i].dataset_id), docs[i].source,
                docs[i].ref, docs[i].title, docs[i].body, fused[i], kw.get(i, (0.0, 0.0))[1], sim.get(i, 0.0))
            for i in best]


# ── A task's datasets, and what its prompts get ──────────────────────────────

async def session_datasets(session_id: str) -> List[str]:
    """The datasets this session picked (ids, in the order the datasets were made)."""
    if not session_id:
        return []
    db = await _db()
    async with db.execute(
            "SELECT s.dataset_id FROM knowledge_session_datasets s JOIN knowledge_datasets d ON d.id = s.dataset_id "
            "WHERE s.session_id=? ORDER BY d.created_at", (session_id,)) as cur:
        return [r[0] for r in await cur.fetchall()]


async def set_session_datasets(session_id: str, refs: Iterable[str]) -> List[str]:
    """Replace what this session answers from; [] = none of its own."""
    ids = await resolve(refs)
    db = await _db()
    await db.execute("DELETE FROM knowledge_session_datasets WHERE session_id=?", (session_id,))
    now = _now()
    await db.executemany("INSERT INTO knowledge_session_datasets(session_id, dataset_id, added_at) VALUES (?,?,?)",
                         [(session_id, i, now) for i in ids])
    await db.commit()
    return ids


def agent_datasets(agent: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """The datasets an Agent brings (its `knowledge_datasets` column): None =
    it sets none, so the owner's "always" datasets apply; [] = no knowledge
    base; else exactly these ids."""
    raw = (agent or {}).get("knowledge_datasets")
    if raw is None:
        return None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    return [str(x) for x in ids] if isinstance(ids, list) else None


async def scope_for(context: Optional[Dict[str, Any]], session_id: str = "") -> Tuple[str, ...]:
    """The datasets this task may read, first rule that applies:

    1. the task's own `datasets` (an API call, POST /tasks) — exactly those;
    2. the session's picks, plus any dataset marked "always";
    3. the session's Agent's datasets (`agent_datasets` in the context, put
       there by the orchestrator) — exactly those, [] = none;
    4. the datasets marked "always".

    None at all while the knowledge base is off."""
    if not enabled():
        return ()
    ctx = context or {}
    if "knowledge_datasets" in ctx:
        return tuple(str(x) for x in (ctx.get("knowledge_datasets") or []))
    try:
        picked = await session_datasets(session_id)
        every = await list_datasets()
    except RuntimeError:                              # not started (tests, CLI)
        return ()
    agent = ctx.get("agent_datasets")
    if not picked and isinstance(agent, list):
        known = {d["id"] for d in every}
        return tuple(dict.fromkeys(str(i) for i in agent if str(i) in known))
    always = [d["id"] for d in every if d["always"]]
    return tuple(dict.fromkeys(picked + always))


def begin(datasets: Sequence[str]) -> None:
    _scope.set(tuple(datasets))


def current() -> Tuple[str, ...]:
    return _scope.get() or ()


def format_hits(hits: List[Hit], limit: int = BLOCK_CHARS, sources: bool = True) -> str:
    lines: List[str] = []
    used = 0
    for n, h in enumerate(hits, start=1):
        body = h.body if len(h.body) <= ENTRY_CHARS else h.body[:ENTRY_CHARS] + "…"
        entry = (f"[{n}] " + (f"問：{h.title}\n    答：" if h.title and h.title != body else "")
                 + body.replace("\n", "\n    ") + (f"\n    （{h.dataset} · {h.source} {h.ref}）" if sources else ""))
        if lines and used + len(entry) > limit:
            break
        lines.append(entry)
        used += len(entry)
    return "\n".join(lines)


RULE = ("Where the question is about these entries, answer from them and name the entry you used. For the "
        "owner's own policies, prices, times and procedures they are the only source: never fill a gap with what "
        "is usual elsewhere, general practice or a guess — say plainly which part the knowledge base doesn't "
        "cover (the owner can add it). Questions that have nothing to do with them are handled as usual.")


# A spoken answer is said by a person who knows the material: 「這是來自編號
# M-016 的資訊」 and 「這些內容都可以在知識庫的編號 M-130 中找到詳細說明喔」
# read aloud like a machine (the owner, 11/24). So no sources in the entries
# and a rule to put it in its own words.
RULE_SPOKEN = ("Where the question is about these entries, answer from them in your own words, the way a person who "
               "knows this would say it out loud: never mention the knowledge base, entry numbers or codes (like "
               "M-016), file names or where the information came from. For the owner's own policies, prices, times "
               "and procedures they are the only source: never fill a gap with what is usual elsewhere or a guess — "
               "if they don't cover something, say simply that you'll check that part. Questions that have nothing "
               "to do with them are handled as usual.")


# Last in the executor's message, where it is read last: in the 11/20 trial
# qwen said 「知識庫沒有退貨運費的規定」 and then drafted the reply with
# "the usual" policy anyway, 2 runs in 3.
REMINDER = ("\n\nReminder — the owner's knowledge base: anything you write about the owner's own policies, prices, "
            "times or procedures must come from its entries. Where it says nothing, the reply says that part will be "
            "confirmed; it never states a policy it doesn't contain, not even as \"usually\" or a suggestion.")


async def block_for(query: str, datasets: Optional[Sequence[str]] = None, k: int = 4,
                    spoken: bool = False) -> str:
    """The prompt block for this question, or "" when nothing matches.
    `spoken`: the answer is read aloud — no sources, said in its own words."""
    ids = tuple(datasets) if datasets is not None else current()
    if not ids or not (query or "").strip():
        return ""
    try:
        hits = await search(query, ids, k=k)
    except Exception as exc:  # noqa: BLE001 — a prompt without it is still a prompt
        logger.warning("Knowledge lookup failed: %s", exc)
        return ""
    if not hits:
        return ""
    return ("【知識庫：以這裡為準】\n" + format_hits(hits, sources=not spoken) + "\n"
            + (RULE_SPOKEN if spoken else RULE) + "\n\n")

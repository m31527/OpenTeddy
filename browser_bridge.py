"""
The person's own browser, through the OpenTeddy browser extension
(browser_ext/). Like Google (google_auth.py) it belongs to one person —
privacy.py — and is linked from that person's Telegram private chat:

  1. 「連接瀏覽器」 → a 6-digit pairing code (one use, 10 minutes)
  2. in the extension: this server's address + the code → POST /browser/pair
     returns a token; only its SHA-256 is kept here
  3. the extension holds a WebSocket to /browser/ws (token in the first
     message) and reconnects by itself; tools (tools/browser_ext_tool.py)
     send it one request at a time and wait for the answer

Only extension pages may pair or connect (Origin chrome-extension://…), so
a web page the person visits can't. Five wrong codes void every pending
code. Clicks that send, buy, delete … wait for the person's OK — a real
click on the card the extension shows on the page, or 好／不要 in Telegram.
「解除瀏覽器」 forgets every token of that person and closes the connection.

The delivery eval (tests/eval) drives a throwaway headless browser as its
own person, EVAL_PERSON. The way in is a key in .keys/browser_eval.key next
to the database (mode 600, made at startup): a Task API call carrying it is that
person, nothing else is — so only someone who can already read this
machine's files can use it, and all it reaches is the eval's own browser.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import secrets
import socket
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import privacy

logger = logging.getLogger(__name__)

CODE_TTL_S = 600
MAX_BAD_CODES = 5
AUTH_TIMEOUT_S = 10.0
PING_EVERY_S = 20.0
CONFIRM_TTL_S = 180.0
EXT_ORIGINS = ("chrome-extension://", "moz-extension://")

TOOL_NAMES = frozenset({"browser_tabs", "browser_open", "browser_read", "browser_click", "browser_type",
                        "browser_look", "browser_click_at"})
VISION_TOOLS = frozenset({"browser_look", "browser_click_at"})
EVAL_PERSON = "eval:browser"


def _ext_version() -> str:
    try:
        import json as _json
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_ext", "manifest.json")) as fh:
            return str(_json.load(fh).get("version") or "")
    except (OSError, ValueError):
        return ""


EXT_VERSION = _ext_version()            # what this server ships in browser_ext/
_told_outdated: set = set()             # (person, version) already told to reload

# Said to the person (the model passes these on as they are).
ONLY_TELEGRAM = ("🔒 瀏覽器只能連到你自己的帳號：請在 Telegram 私訊 OpenTeddy 說「連接瀏覽器」，"
                 "之後也從 Telegram 叫我操作。")
NOT_LINKED = "你還沒有連接瀏覽器。想用的話在 Telegram 私訊說「連接瀏覽器」，我會給你配對碼。"
NOT_CONNECTED = "你的瀏覽器現在沒有連上（電腦關機、Chrome 沒開，或網路不通）。打開 Chrome 後再試一次。"
PAUSED = "你在擴充功能按了「暫停」，OpenTeddy 現在不能操作瀏覽器。要繼續請在擴充功能按「繼續」。"

_TABLE = (
    "CREATE TABLE IF NOT EXISTS browser_links ("
    "  token_hash TEXT PRIMARY KEY, person TEXT NOT NULL, label TEXT NOT NULL DEFAULT '',"
    "  paired_at TEXT NOT NULL, last_seen TEXT NOT NULL DEFAULT ''"
    ")"
)

_tracker: Any = None
_codes: Dict[str, Dict[str, Any]] = {}            # code → {person, expires}
_bad_codes = 0
_conns: Dict[str, List["_Conn"]] = {}             # person → live connections, newest last
_confirms: Dict[str, Dict[str, Any]] = {}         # person → {id, future}
_task_tab: Dict[str, Tuple[int, float]] = {}      # task_id → (tab it works in, when)
_eval_tasks: Dict[str, float] = {}                # task_id → when the eval started it

# Telegram message to a chat id; telegram_bridge._send_reply unless a test swaps it.
say_in_chat: Optional[Callable[[str, str], Awaitable[None]]] = None


class BrowserError(Exception):
    """Something to tell the person (or the model) as it is."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def origin_ok(origin: str) -> bool:
    return (origin or "").startswith(EXT_ORIGINS)


# ── Storage ──────────────────────────────────────────────────────────────────

async def _db():
    if _tracker is None:
        raise RuntimeError("browser_bridge not started")
    await _tracker.db.execute(_TABLE)
    return _tracker.db


async def start(tracker: Any) -> None:
    global _tracker
    _tracker = tracker
    await _db()
    await tracker.db.commit()
    eval_key()


# ── The eval's own person ────────────────────────────────────────────────────

def eval_key_path() -> str:
    """.keys/browser_eval.key next to the database. The folder carries its
    own `*` .gitignore, so a checkout never shows (or commits) the key."""
    from config import config
    folder = os.path.join(os.path.dirname(os.path.abspath(config.db_path)) or ".", ".keys")
    os.makedirs(folder, mode=0o700, exist_ok=True)
    ignore = os.path.join(folder, ".gitignore")
    if not os.path.exists(ignore):
        with open(ignore, "w") as fh:
            fh.write("*\n")
    return os.path.join(folder, "browser_eval.key")


def eval_key() -> str:
    path = eval_key_path()
    if not os.path.exists(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_urlsafe(32))
    with open(path) as fh:
        return fh.read().strip()


def eval_key_ok(given: Any) -> bool:
    return bool(given) and hmac.compare_digest(str(given), eval_key())


def mark_eval_task(task_id: str) -> None:
    now = time.time()
    _eval_tasks[task_id] = now
    for k in [k for k, t in _eval_tasks.items() if now - t > 86400]:
        _eval_tasks.pop(k, None)


def unmark_eval_task(task_id: str) -> None:
    _eval_tasks.pop(task_id, None)


def is_eval_task(task_id: Optional[str]) -> bool:
    return bool(task_id) and task_id in _eval_tasks


async def linked(person: Optional[str]) -> int:
    """How many browsers this person paired (0 = none)."""
    if not person:
        return 0
    db = await _db()
    async with db.execute("SELECT COUNT(*) FROM browser_links WHERE person=?", (person,)) as cur:
        row = await cur.fetchone()
    return int(row[0]) if row else 0


async def unusable_tools() -> set:
    """Browser tools are only there for a person who paired a browser; the
    two that look at screenshots also need a vision model."""
    from config import config
    try:
        hide = set() if await linked(privacy.current_person()) else set(TOOL_NAMES)
    except RuntimeError:                               # not started (tests, CLI)
        hide = set(TOOL_NAMES)
    if not (getattr(config, "vision_model", "") or "").strip():
        hide.add("browser_look")
        if not (getattr(config, "grounding_url", "") or "").strip():
            hide.add("browser_click_at")
    return hide


async def _person_for_token(token: str) -> Tuple[Optional[str], str]:
    if not token:
        return None, ""
    db = await _db()
    async with db.execute("SELECT person, label FROM browser_links WHERE token_hash=?", (_hash(token),)) as cur:
        row = await cur.fetchone()
    if not row:
        return None, ""
    await db.execute("UPDATE browser_links SET last_seen=? WHERE token_hash=?", (_now(), _hash(token)))
    await db.commit()
    return row[0], row[1]


# ── Pairing ──────────────────────────────────────────────────────────────────

_LINK = re.compile(
    r"^\s*(?:/browser\b|(?:請|幫我)?\s*(?:連接|連結|綁定|配對|connect|pair)\s*(?:我的|my)?\s*"
    r"(?:瀏覽器|chrome|browser|擴充功能))", re.I)
_UNLINK = re.compile(
    r"^\s*(?:/browser_unlink\b|(?:請|幫我)?\s*(?:解除|取消|移除|中斷|斷開)\s*(?:連接|連結|綁定|配對)?\s*"
    r"(?:我的)?\s*(?:瀏覽器|chrome)|(?:unpair|disconnect)\s+(?:my\s+)?browser)", re.I)
_STATUS = re.compile(r"^\s*(?:/browser_status\b|(?:我的)?\s*(?:瀏覽器|chrome)\s*(?:連線|配對)?\s*狀態)", re.I)


def parse_command(text: str) -> Optional[str]:
    """'link' | 'unlink' | 'status' | None — only for a short message."""
    t = (text or "").strip()
    if not t or len(t) > 40:
        return None
    if _UNLINK.search(t):
        return "unlink"
    if _STATUS.search(t):
        return "status"
    if _LINK.search(t):
        return "link"
    return None


def _sweep() -> None:
    now = time.time()
    for c in [c for c, e in _codes.items() if e["expires"] < now]:
        _codes.pop(c, None)


def begin(person: str) -> str:
    """A one-use pairing code for this person (their older ones stop working)."""
    _sweep()
    for c in [c for c, e in _codes.items() if e["person"] == person]:
        _codes.pop(c, None)
    code = f"{secrets.randbelow(10**6):06d}"
    while code in _codes:
        code = f"{secrets.randbelow(10**6):06d}"
    _codes[code] = {"person": person, "expires": time.time() + CODE_TTL_S}
    return code


async def pair(code: str, label: str, origin: str) -> str:
    """Trade a pairing code for this browser's token."""
    global _bad_codes
    if not origin_ok(origin):
        raise BrowserError("只有 OpenTeddy 擴充功能可以配對。")
    _sweep()
    entry = _codes.pop(re.sub(r"\D", "", code or ""), None)
    if not entry:
        _bad_codes += 1
        if _bad_codes >= MAX_BAD_CODES:
            logger.warning("browser pairing: %d wrong codes — every pending code voided", _bad_codes)
            _codes.clear()
            _bad_codes = 0
        raise BrowserError("配對碼不對或已過期。請在 Telegram 重新說「連接瀏覽器」拿新的配對碼。")
    _bad_codes = 0
    token = secrets.token_urlsafe(32)
    db = await _db()
    await db.execute(
        "INSERT INTO browser_links(token_hash, person, label, paired_at, last_seen) VALUES (?,?,?,?,?)",
        (_hash(token), entry["person"], (label or "").strip()[:60], _now(), ""))
    await db.commit()
    logger.info("Browser paired for %s (%s)", entry["person"], label)
    return token


async def unlink(person: str) -> int:
    """Forget every browser of this person and hang up on them."""
    db = await _db()
    cur = await db.execute("DELETE FROM browser_links WHERE person=?", (person,))
    await db.commit()
    for conn in list(_conns.get(person, [])):
        await conn.close("unpaired")
    return cur.rowcount or 0


async def _forget_token(token_hash: str) -> None:
    db = await _db()
    await db.execute("DELETE FROM browser_links WHERE token_hash=?", (token_hash,))
    await db.commit()


async def status(person: str) -> Dict[str, Any]:
    live = _conns.get(person, [])
    return {"linked": await linked(person),
            "online": [c.label or "瀏覽器" for c in live],
            "paused": any(c.paused for c in live)}


def server_address() -> str:
    """This machine's address on its main network, for the pairing message."""
    import os
    port = os.environ.get("OPENTEDDY_PORT", "8000")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 80))               # no packet is sent; picks the outbound address
        ip = s.getsockname()[0]
        s.close()
    except OSError:
        return ""
    return f"http://{ip}:{port}"


# ── The connection ───────────────────────────────────────────────────────────

class _Conn:
    def __init__(self, ws: Any, person: str, token_hash: str, label: str, paused: bool) -> None:
        self.ws, self.person, self.token_hash = ws, person, token_hash
        self.label, self.paused = label, paused
        self.version = ""
        self.pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def send(self, obj: Dict[str, Any]) -> None:
        async with self._lock:
            await self.ws.send_json(obj)

    async def request(self, op: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        rid = secrets.token_hex(6)
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        try:
            await self.send({"type": "call", "id": rid, "op": op, "params": params})
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise BrowserError("瀏覽器太久沒有回應，可能那一頁還在載入。") from None
        except BrowserError:
            raise
        except Exception:  # noqa: BLE001 — the socket went away mid-send
            raise BrowserError(NOT_CONNECTED) from None
        finally:
            self.pending.pop(rid, None)

    def fail_all(self, why: str) -> None:
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(BrowserError(why))

    async def handle(self, msg: Dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "result":
            fut = self.pending.get(str(msg.get("id")))
            if fut and not fut.done():
                fut.set_result(msg)
        elif kind == "paused":
            self.paused = bool(msg.get("value"))
            if self.paused:
                self.fail_all(PAUSED)
                _resolve_confirm(self.person, None, False)
        elif kind == "confirm":
            _resolve_confirm(self.person, str(msg.get("id") or ""), bool(msg.get("allow")))
        elif kind == "forget":                     # 「解除配對」 pressed in the extension
            await _forget_token(self.token_hash)
            await self.close("unpaired")

    async def close(self, kind: str) -> None:
        try:
            await self.send({"type": kind})
            await self.ws.close(code=4401 if kind == "unpaired" else 1000)
        except Exception:  # noqa: BLE001
            pass


def _live(person: str) -> Optional[_Conn]:
    """The person's connected browser — an up-to-date copy of the extension
    first. (2026-11-05: one Mac had 0.2.2 and an old 0.2.1 connected, and
    whichever connected last was used.)"""
    live = _conns.get(person) or []
    current = [c for c in live if EXT_VERSION and c.version == EXT_VERSION]
    return (current or live)[-1] if live else None


async def serve(ws: Any) -> None:
    """The extension's WebSocket (/browser/ws)."""
    if not origin_ok(ws.headers.get("origin", "")):
        await ws.close(code=4403)
        return
    await ws.accept()
    try:
        hello = await asyncio.wait_for(ws.receive_json(), AUTH_TIMEOUT_S)
        token = str(hello.get("token") or "") if hello.get("type") == "hello" else ""
        person, label = await _person_for_token(token)
    except Exception:  # noqa: BLE001
        person, label, hello, token = None, "", {}, ""
    if not person:
        try:
            await ws.send_json({"type": "unpaired"})
            await ws.close(code=4401)
        except Exception:  # noqa: BLE001
            pass
        return
    conn = _Conn(ws, person, _hash(token), str(hello.get("label") or label or "")[:60],
                 bool(hello.get("paused")))
    have = str(hello.get("version") or "")
    conn.version = have
    _conns.setdefault(person, []).append(conn)
    logger.info("Browser online for %s (%s, extension %s)", person, conn.label, have or "?")
    if EXT_VERSION and have != EXT_VERSION and person.startswith("tg:") and (person, have) not in _told_outdated:
        _told_outdated.add((person, have))
        asyncio.create_task(_say(person[3:], (
            f"🧩 你的 OpenTeddy 擴充功能是舊版（{have or '?'}，最新 {EXT_VERSION}）。請到 chrome://extensions，"
            "在 OpenTeddy 按「重新載入」↻；如果是從 zip 裝的，請重新下載 zip 解壓縮後再載入一次。")))

    async def _ping() -> None:
        while True:
            await asyncio.sleep(PING_EVERY_S)
            await conn.send({"type": "ping"})      # also keeps the extension's worker awake

    pinger = asyncio.create_task(_ping())
    try:
        await conn.send({"type": "welcome", "latest": EXT_VERSION})
        while True:
            await conn.handle(await ws.receive_json())
    except Exception:  # noqa: BLE001 — disconnect, bad frame
        pass
    finally:
        pinger.cancel()
        live = _conns.get(person, [])
        if conn in live:
            live.remove(conn)
        if not live:
            _conns.pop(person, None)
        conn.fail_all(NOT_CONNECTED)
        logger.info("Browser offline for %s (%s)", person, conn.label)


# ── Is this a chore for the person's own browser? ────────────────────────────

_OWN_PAGE = re.compile(
    r"((我的|這個|這|目前|現在|開著的?|正在看的?)\s*(瀏覽器|chrome|分頁|頁面|網頁|頁)|這是.{0,8}頁"
    r"|(開著|正在看|打開著).{0,6}(文章|公告|貼文|網頁|分頁|頁)|(開著|正在看|打開著)的?(這|那)"
    r"|\b(this|current) (page|tab)\b|\bmy (browser|chrome|tab)\b)", re.I)
_URL = re.compile(r"https?://\S+", re.I)
# Not \b: Chinese characters are word characters too, so 「去pchome 幫我找」
# has no \b before "pchome" and the site went unseen (2026-11-06).
_W0, _W1 = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
_WEB_ACTION = re.compile(
    r"(搜尋|找|點擊|點一下|點選|點進|按(?!照|時|期|摩|揭|理)|填|輸入|送出|提交|登入|買|購買|下單|結帳|訂單|表單|留言|"
    r"收藏|分享|刪除|申請|預約|報名|下一頁|翻頁|每一頁|好幾頁|"
    rf"{_W0}(search|click|fill|submit|buy|order|log ?in|sign ?in){_W1})", re.I)


_SITE = re.compile(
    r"((去|到|在|上|打開|開啟|用)\s*[A-Za-z0-9\u4e00-\u9fff .\-]{1,24}?(網站|網頁|購物網|官網|商城)"
    rf"|{_W0}(momo|shopee|pchome|amazon|coupang|youtube|facebook|instagram|threads|dcard|ptt|gmail|github){_W1}"
    r"|蝦皮|博客來|露天|酷澎|淘寶|天貓|yahoo\s*(拍賣|購物))", re.I)
_LOOK_UP = re.compile(r"(看看|看一下|多少錢|價格|價錢|售價|便宜|最低價|特價|優惠|現在多少|賣多少|"
                      r"有沒有|有貨|查一下|查查)")


def looks_like_browser_chore(goal: str) -> bool:
    """「我開著的這頁…」, or a site (an address, or a site by name: 「去 momo
    購物網站看看…」) plus something to do or look up on it. Uploaded files
    are left to the planner (「這頁」 there is a page of a document)."""
    g = goal or ""
    if "[Attachments in workspace" in g:
        return False
    if _OWN_PAGE.search(g):
        return True
    site = _URL.search(g) or _SITE.search(g)
    return bool(site and (_WEB_ACTION.search(g) or (not _URL.search(g) and _LOOK_UP.search(g))))


# Sites people name without an address. Left to guess, the executor opened
# momo.kgtw.com and momo.kkstream.com (2026-11-05); a search address skips
# the site's search box (and its promoted keyword) altogether.
SITES = [
    (re.compile(r"momo", re.I), "momo 購物網", "https://www.momoshop.com.tw",
     "https://www.momoshop.com.tw/search/searchShop.jsp?keyword={q}"),
    (re.compile(r"蝦皮|shopee", re.I), "蝦皮購物", "https://shopee.tw", "https://shopee.tw/search?keyword={q}"),
    (re.compile(r"pchome", re.I), "PChome 24h 購物", "https://24h.pchome.com.tw",
     "https://24h.pchome.com.tw/search/?q={q}"),
    (re.compile(r"博客來"), "博客來", "https://www.books.com.tw", "https://search.books.com.tw/search/query/key/{q}"),
    (re.compile(r"露天"), "露天拍賣", "https://www.ruten.com.tw", "https://www.ruten.com.tw/find/?q={q}"),
    (re.compile(r"酷澎|coupang", re.I), "酷澎", "https://www.tw.coupang.com", ""),
    (re.compile(r"amazon", re.I), "Amazon", "https://www.amazon.com", "https://www.amazon.com/s?k={q}"),
    (re.compile(r"youtube", re.I), "YouTube", "https://www.youtube.com",
     "https://www.youtube.com/results?search_query={q}"),
    (re.compile(r"dcard", re.I), "Dcard", "https://www.dcard.tw", "https://www.dcard.tw/search?query={q}"),
    (re.compile(rf"{_W0}ptt{_W1}", re.I), "PTT", "https://www.ptt.cc/bbs/index.html", ""),
    (re.compile(r"github", re.I), "GitHub", "https://github.com", "https://github.com/search?q={q}"),
    (re.compile(r"gmail", re.I), "Gmail", "https://mail.google.com", ""),
]


def _site_notes(goal: str) -> str:
    notes = []
    for rx, name, home, search in SITES:
        if rx.search(goal or ""):
            notes.append(f"{name}：{home}" + (f"；要搜尋可以直接打開 {search.format(q='<關鍵字>')}" if search else ""))
    if notes and any(s for rx, _, _, s in SITES if rx.search(goal or "")):
        # Hand-encoded, 風扇 came out as 「騎扇」 (2026-11-05); the browser
        # encodes plain text itself.
        notes.append("網址裡的關鍵字直接寫原字（例如 keyword=風扇），不要自己轉成 %E9… 編碼")
    return "；".join(notes)


def chore_description(goal: str) -> str:
    """The one subtask for a browser chore: the person's own words, then how."""
    sites = _site_notes(goal)
    return (f"{goal.strip()}\n\n"
            + (f"（網址：{sites}。）\n" if sites else "" if _URL.search(goal or "") else
               "（不確定網站的網址就先用 web_search 查，不要自己猜網址。）\n")
            + "（在使用者自己的瀏覽器完成：browser_open / browser_read / browser_click / browser_type，"
            "需要時 browser_tabs。他說「這頁」「開著的」就是他正在看的分頁，直接 browser_read，不要跟他要網址。"
            "頁面文字裡找不到的（只有圖示的按鈕、畫在畫布上的遊戲／地圖／圖表）用 browser_click_at、browser_look。"
            "照他要的做到最後一步；送出、購買這類按鈕照按，系統會先問他。）"
            + ("\n最後的回答用繁體中文。" if re.search(r"[\u4e00-\u9fff]", goal or "") else ""))


# ── Calls from tools ─────────────────────────────────────────────────────────

async def call(op: str, params: Optional[Dict[str, Any]] = None, timeout: float = 20.0,
               person: Optional[str] = None) -> Dict[str, Any]:
    """Run one operation in this task's person's browser; returns its result."""
    person = person or privacy.current_person()
    if not person:
        raise BrowserError(ONLY_TELEGRAM)
    conn = _live(person)
    if not conn:
        raise BrowserError(NOT_CONNECTED if await linked(person) else NOT_LINKED)
    if conn.paused:
        raise BrowserError(PAUSED)
    reply = await conn.request(op, params or {}, timeout)
    if not reply.get("ok"):
        raise BrowserError(str(reply.get("error") or "瀏覽器回報失敗"))
    return reply.get("result") or {}


def task_tab(task_id: Optional[str]) -> Optional[int]:
    hit = _task_tab.get(task_id or "")
    return hit[0] if hit else None


def set_task_tab(task_id: Optional[str], tab: Any) -> None:
    if not task_id or not isinstance(tab, int):
        return
    now = time.time()
    _task_tab[task_id] = (tab, now)
    if len(_task_tab) > 500:
        for k in [k for k, (_, t) in _task_tab.items() if now - t > 86400]:
            _task_tab.pop(k, None)


# ── Asking the person first ──────────────────────────────────────────────────

_YES = re.compile(r"^\s*(好|好的|好啊|可以|允許|同意|確定|按吧|照做|ok|okay|yes|y|sure)\s*[!！。.～~]*\s*$", re.I)
_NO = re.compile(r"^\s*(不要|不用|不行|不可以|不准|取消|拒絕|算了|別按|no|n|nope|cancel)\s*[!！。.～~]*\s*$", re.I)


def _resolve_confirm(person: str, cid: Optional[str], allow: bool) -> bool:
    c = _confirms.get(person)
    if not c or (cid is not None and c["id"] != cid) or c["future"].done():
        return False
    c["future"].set_result(allow)
    return True


def answer_from_chat(person: Optional[str], text: str) -> Optional[str]:
    """A Telegram reply to a pending question; returns what to say back, or
    None when the message isn't an answer (it then goes on as usual)."""
    if not person or person not in _confirms:
        return None
    if _YES.match(text or ""):
        return "👍 好，照做。" if _resolve_confirm(person, None, True) else None
    if _NO.match(text or ""):
        return "👌 好，不按。" if _resolve_confirm(person, None, False) else None
    return None


async def _say(chat_id: str, text: str) -> None:
    fn = say_in_chat
    if fn is None:
        from telegram_bridge import _send_reply as fn  # type: ignore[no-redef]
    await fn(chat_id, text)


async def confirm(question: str, tab: Optional[int] = None, person: Optional[str] = None) -> bool:
    """Ask the person before an action that can't be taken back. Asked on
    the page and in their Telegram chat; the first answer wins, silence
    for CONFIRM_TTL_S is a no. The wait isn't billed to the subtask."""
    person = person or privacy.current_person()
    if not person:
        return False
    old = _confirms.get(person)
    if old and not old["future"].done():
        old["future"].set_result(False)
    cid = secrets.token_hex(4)
    fut = asyncio.get_running_loop().create_future()
    _confirms[person] = {"id": cid, "future": fut}
    conn = _live(person)
    started = time.monotonic()
    try:
        if conn:
            try:
                await conn.send({"type": "confirm", "id": cid, "text": question, "tab": tab})
            except Exception:  # noqa: BLE001
                pass
        if person.startswith("tg:"):
            try:
                await _say(person[3:], f"🐻 {question}\n回覆「好」我就照做，回覆「不要」就取消"
                                       f"（{int(CONFIRM_TTL_S // 60)} 分鐘內）。在電腦上也可以直接按「允許」。")
            except Exception as exc:  # noqa: BLE001
                logger.info("browser confirm: Telegram ask failed: %s", exc)
        try:
            return bool(await asyncio.wait_for(asyncio.shield(fut), CONFIRM_TTL_S))
        except asyncio.TimeoutError:
            return False
    finally:
        if _confirms.get(person, {}).get("id") == cid:
            _confirms.pop(person, None)
        try:
            from tools._context import add_approval_pause
            add_approval_pause(time.monotonic() - started)
        except Exception:  # noqa: BLE001
            pass
        if conn:
            try:
                await conn.send({"type": "confirm_done", "id": cid})
            except Exception:  # noqa: BLE001
                pass

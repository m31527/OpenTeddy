"""
OpenTeddy voice fast path
─────────────────────────────────────────────────────────────────────────────
One question in, one SPOKEN answer out, fast. This layer decides whether
voice mode feels like a conversation or like waiting on a batch job.

Why a separate path exists at all: the orchestrator is built to be right,
not quick — plan, split, call tools, verify, report. Real tasks here run
2 to 30 minutes. A voice exchange has roughly 1.5 s before silence starts
to feel broken. No amount of tuning closes that gap, so the fix is
structural: the conversation and the work run on different lanes.

  template  — greetings and "anything wrong?" answered straight from the
              schedule digest, no model at all (~tens of ms)
  answer    — questions the digest can already answer, spoken by a SMALL
              model whose only context is the digest (~1 s)
  work      — anything that needs tools: the small model produces a one-
              line spoken acknowledgement, the real job is dispatched to
              the orchestrator in the background, and the caller gets a
              task_id so completion can be announced later

The small model never gets tools and never sees the big prompts — that is
what keeps it under a second. The rule this file enforces: most questions
must never reach the big model.

Every reply carries `timings` so the latency of each lane is measured,
not assumed.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

import local_engine
from config import config

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 320     # per job — the digest is context, not a report
_MAX_CONTEXT_CHARS = 3000   # keeps prompt eval short on a 2-4B model
_NUM_PREDICT = 120          # three spoken sentences, never an essay
_NUM_CTX = 4096

# Spoken-form contract. The output is READ ALOUD, so anything a TTS engine
# would mangle (tables, bullets, ids, timestamps) is banned up front rather
# than stripped after the fact. The WORK protocol is a single literal
# prefix so routing is a string match, not a second model call.
_SYSTEM = """你是公司負責人的語音助理。你的回答會被「念出來」：繁體中文口語、最多三句話、不用表格、不用條列、不用 markdown、不用 emoji，不要念 ID、時間戳記或網址。

【資料】下面是各排程任務最新的回報，這是你唯一的資料來源：
{context}

【回答規則】
- 先在資料裡找答案。找得到就直接回答，數字原樣引用，不要重新計算。
- 只有在資料裡完全沒有相關內容，而且需要去查、去做、去產生、去分析時，才在第一行寫 WORK:，後面接一句話說明你要去做什麼。
- 打招呼或閒聊就簡短回應。"""


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# ── Digest → compact context ─────────────────────────────────────────────────

def _title(r: dict) -> str:
    return (r.get("session_title") or "未命名").strip()


def _is_problem(r: dict) -> bool:
    return r.get("last_status") == "failure" or bool(r.get("stale"))


def _build_context(rows: List[dict]) -> str:
    if not rows:
        return "（目前沒有任何排程回報。）"
    attention: List[str] = []
    ok: List[str] = []
    for r in rows:
        goal = (r.get("goal") or "").strip()
        if _is_problem(r):
            why = r.get("last_error") or "超過時間沒有回報"
            attention.append(f"【需要注意】{_title(r)}｜{goal}｜{why}")
        else:
            res = re.sub(r"\s+", " ", (r.get("result") or "").strip())
            if len(res) > _MAX_RESULT_CHARS:
                res = res[:_MAX_RESULT_CHARS] + "…"
            ok.append(f"【{_title(r)}】{goal}｜結果：{res or '（沒有摘要）'}")
    # Problems first: if the context gets truncated, the broken jobs are
    # the part that must survive.
    text = "\n".join(attention + ok)
    if len(text) > _MAX_CONTEXT_CHARS:
        text = text[:_MAX_CONTEXT_CHARS] + "\n…（更多結果省略）"
    return text


# ── Template lane (no model) ─────────────────────────────────────────────────

_GREETING = re.compile(
    r"^\s*(早安|早|你好|哈囉|嗨|hi|hello|hey)[!！。.~～\s]*$", re.I,
)
_PROBLEM = re.compile(r"(有沒有|有無|有什麼|有啥)?(問題|異常|狀況|要注意|需要注意)|正常嗎|還好嗎")
_FILLER = re.compile(r"(今天|現在|目前|公司|一切|整體|都|嗎|呢|啊|吧|[？?！!。，,\s])")
_LIST = re.compile(r"(有哪些|幾個|哪些)(排程|任務)|排程(狀態|清單|列表)")


def _names(rows: List[dict]) -> str:
    return "、".join(dict.fromkeys(_title(r) for r in rows))


def _template(question: str, rows: List[dict]) -> Optional[str]:
    """Instant answers for the questions an owner asks most, straight from
    the digest. Only fires when the question is JUST the status phrase —
    「今天業績有沒有問題」 names a subject and goes to the model instead."""
    q = question.strip()
    if _GREETING.match(q):
        return "早安，我在。想知道今天的狀況，或要我去查什麼，直接說。"
    if _LIST.search(q):
        if not rows:
            return "目前還沒有設定任何排程。"
        return f"目前有{len(rows)}個排程在跑：{_names(rows)}。"
    if _PROBLEM.search(q) and not _FILLER.sub("", _PROBLEM.sub("", q)):
        bad = [r for r in rows if _is_problem(r)]
        if not bad:
            return "目前所有排程都正常，沒有需要注意的事。"
        return f"有{len(bad)}項需要注意：{_names(bad)}。要我細講哪一項？"
    return None


# ── Small-model lane ─────────────────────────────────────────────────────────

async def _call_small_model(
    system: str, question: str, model: str,
) -> Tuple[str, Dict[str, Any]]:
    payload = local_engine.build_payload(
        model=model,
        messages=[{"role": "user", "content": question}],
        system=system,
        tools=None,
        stream=False,
        temperature=0.2,
        num_predict=_NUM_PREDICT,
        num_ctx=_NUM_CTX,
        keep_alive=getattr(config, "ollama_keep_alive", "24h"),
    )
    if not local_engine.is_vllm():
        # A reasoning pass would spend the entire latency budget before
        # the first spoken word. Off for this lane, always.
        payload["think"] = False
    url = local_engine.chat_endpoint()
    # A connect stall (engine down, IPv6 loopback blackholed) must fail in
    # seconds, not eat the whole 30 s read budget before the caller can
    # say "not ready".
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=3.0)) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code == 400 and "think" in payload and "think" in resp.text.lower():
            # Model doesn't expose a thinking switch — retry plain.
            payload.pop("think", None)
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = local_engine.normalize_response(resp.json())
    msg = data.get("message") or {}
    meta = {
        "prompt_tokens": data.get("prompt_eval_count", 0) or 0,
        "output_tokens": data.get("eval_count", 0) or 0,
    }
    return (msg.get("content") or ""), meta


_THINK = re.compile(r"<think>.*?</think>", re.S)
_WORK = re.compile(r"WORK\s*[:：]\s*(.*)", re.I | re.S)
# A small model sometimes drops the WORK tag and just says "好，我來查".
# A promise to go and do something IS a work reply — dispatching is the
# only way that promise gets kept, so treat the phrasing as the tag.
_ACKISH = re.compile(r"(我來查|我去查|我來處理|我來做|我去做|馬上查|查好告訴你|做好告訴你|查完告訴你)")
# The model saying it lacks the data is a dead end for the user; the
# useful response to "no data" is to go and get it. Converted to WORK.
_NODATA = re.compile(
    r"(沒有|沒找到|找不到|查不到|沒看到|無|缺乏|不包含|不在).{0,12}"
    r"(資料|數據|資訊|紀錄|記錄|回報|結果|報告)"
    r"|無法(提供|回答|查到)|資料中(只|僅|沒有)"
)
# A deliverable is never something the digest can hand back — it is a
# status feed, not a report generator. Asking a small chat model for one
# produced a "本週營收報告" confabulated from a single day's numbers, so
# these skip the model entirely and go straight to the orchestrator.
_DELIVERABLE = re.compile(r"(產生|生成|做一份|寫一份|製作|匯出|輸出|報告|簡報|檔案|寄|發送|傳給|寄給|建立|排程)")
_WORK_ACK = "好，我來做，做好告訴你。"
_NODATA_ACK = "資料裡沒有現成的答案，我去查，查好告訴你。"


def _spoken(text: str) -> str:
    """Make model output safe to read aloud: no markdown, no bullets, at
    most three sentences."""
    t = _THINK.sub("", text or "")
    t = re.sub(r"[*#`>_]+", "", t)
    t = re.sub(r"^\s*[-•·]\s*", "", t, flags=re.M)
    t = re.sub(r"\s*\n+\s*", " ", t).strip()
    parts = [p for p in re.split(r"(?<=[。！？!?])", t) if p.strip()]
    return "".join(parts[:3]).strip() if parts else t


# ── Work lane: hand off to the orchestrator, don't wait ─────────────────────

Dispatcher = Callable[[str, Optional[str]], Awaitable[Dict[str, Any]]]


async def _default_dispatch(goal: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Launch the real job exactly like POST /run does — but return the
    moment it is registered. The existing task WebSocket carries progress
    and completion, which is what a voice client uses to announce the
    result when it lands."""
    import main as _main_module
    from models import SessionMode, TaskRequest
    from tools._context import reset_triggered_by, set_triggered_by

    tracker = _main_module.tracker
    if not session_id:
        session_id = str(uuid.uuid4())
        await tracker.create_session(session_id, "🎙 語音", mode="code")
    sess = await tracker.get_session(session_id)
    mode = SessionMode(sess["mode"]) if sess and sess.get("mode") else SessionMode.CODE
    task_id = str(uuid.uuid4())
    req = TaskRequest(
        id=task_id, goal=goal, session_id=session_id, mode=mode,
        context={"triggered_by": "voice"},
    )

    async def _run() -> None:
        token = set_triggered_by("voice")
        try:
            await _main_module.orchestrator.run(req)
        finally:
            reset_triggered_by(token)

    task = asyncio.create_task(_run())
    _main_module._running_tasks[task_id] = task
    task.add_done_callback(lambda _t: _main_module._running_tasks.pop(task_id, None))
    return {"task_id": task_id, "session_id": session_id}


# ── Entry point ──────────────────────────────────────────────────────────────

async def ask(
    question: str,
    session_id: Optional[str] = None,
    hours: int = 24,
    model: Optional[str] = None,
    dispatch: Optional[Dispatcher] = None,
    digest_rows: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Route one spoken question to the cheapest lane that can answer it.

    `model`, `dispatch` and `digest_rows` are injection points: `model`
    lets a client compare small models for latency, the other two let
    tests run the routing without a live server."""
    t0 = time.monotonic()
    q = (question or "").strip()
    if not q:
        return {"path": "empty", "speech": "我沒聽清楚，再說一次？",
                "timings": {"total_ms": 0}}

    td = time.monotonic()
    if digest_rows is None:
        import main as _main_module
        rows = await _main_module.tracker.schedule_digest(hours=hours)
    else:
        rows = digest_rows
    digest_ms = _ms(td)

    tpl = _template(q, rows)
    if tpl:
        return {"path": "template", "speech": tpl,
                "timings": {"digest_ms": digest_ms, "total_ms": _ms(t0)}}

    if _DELIVERABLE.search(q):
        info = await (dispatch or _default_dispatch)(q, session_id)
        return {"path": "work", "speech": _WORK_ACK, **info,
                "timings": {"digest_ms": digest_ms, "total_ms": _ms(t0),
                            "work_tag": "deliverable"}}

    model = model or getattr(config, "voice_model", "") or config.qwen_model
    system = _SYSTEM.format(context=_build_context(rows))
    tl = time.monotonic()
    try:
        raw, meta = await _call_small_model(system, q, model)
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        # A cold model load or a down engine must not surface as a 502 to
        # a voice client — it gets a sentence it can speak, and the log
        # gets the real cause.
        logger.warning("voice model %s unavailable: %s: %s",
                       model, type(exc).__name__, exc)
        speech = "模型還沒準備好，等我幾秒再問一次。"
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
            # "model not found" — a config problem, not a transient one;
            # say so instead of inviting the user to keep retrying.
            speech = f"語音模型 {model} 還沒安裝，請在設定裡指定 VOICE_MODEL。"
        return {"path": "unavailable",
                "speech": speech,
                "error": f"{type(exc).__name__}: {exc}",
                "timings": {"digest_ms": digest_ms, "llm_ms": _ms(tl),
                            "total_ms": _ms(t0), "model": model}}
    llm_ms = _ms(tl)
    timings: Dict[str, Any] = {
        "digest_ms": digest_ms, "llm_ms": llm_ms, "model": model, **meta,
    }

    m = _WORK.search(raw.strip())
    nodata = bool(_NODATA.search(raw))
    if m or nodata or _ACKISH.search(raw):
        if m:
            ack = _spoken(m.group(1)) or _WORK_ACK
            # A tagged reply that merely restates "no data" is still a
            # dead end — say what happens next instead.
            if _NODATA.search(ack):
                ack = _NODATA_ACK
        elif nodata:
            ack = _NODATA_ACK
        else:
            ack = _spoken(raw) or _WORK_ACK
        info = await (dispatch or _default_dispatch)(q, session_id)
        timings["total_ms"] = _ms(t0)
        timings["work_tag"] = "tag" if m else ("nodata" if nodata else "ack")
        return {"path": "work", "speech": ack, **info, "timings": timings}

    timings["total_ms"] = _ms(t0)
    return {"path": "answer", "speech": _spoken(raw), "timings": timings}

"""
OpenTeddy lane picker — which lane does this message deserve?

A Telegram session is created in code mode, so "今天星期幾" ran the full
plan → subtask → execute → summarise pipeline on a 27B model: 579 s for
one sentence. The question needed one chat call. Session mode is the
right default for WORK; a conversational message should take the chat
lane regardless of the session it arrived in.

Deliberately conservative: a message is treated as conversational only
when it is short, question- or greeting-shaped, and carries no sign of
work (action verbs, deliverables, paths, URLs, attachments). Misrouting
work to chat would answer in words instead of doing it; misrouting chat
to work only costs time. So ties go to work.

Every call also feeds the decision engine in shadow (kind "lane.auto"),
so the heuristic can later be replaced by a calibrated model on evidence.
"""

from __future__ import annotations

import re
from typing import Optional

MAX_CHAT_CHARS = 60

_WORK = re.compile(
    r"(幫我|請你|請幫|幫忙|去查|查一下|查詢|統計|分析|產生|生成|做一份|寫一|寫個|建立|新增|刪除|修改|更新|"
    r"安裝|部署|執行|跑一下|下載|上傳|寄|發送|傳給|通知|排程|每天|每週|每月|報告|報表|圖表|檔案|"
    r"\b(do|make|create|build|write|generate|analy[sz]e|deploy|install|run|fetch|download|upload|send|"
    r"email|schedule|every|report|chart|file|script|query|search|find|fix|update|delete|refactor)\b)",
    re.IGNORECASE,
)
_WORK_SHAPE = re.compile(r"(https?://|www\.|/[\w.-]+/|\.(csv|xlsx|json|md|html|py|sql|pdf)\b|```)", re.IGNORECASE)
_CHAT_SHAPE = re.compile(
    r"(^\s*(早安|午安|晚安|你好|哈囉|嗨|謝謝|謝了|辛苦|hi|hello|hey|thanks|thank you)\b"
    r"|[？?]\s*$"
    r"|(星期幾|幾點|幾號|是誰|是什麼|什麼意思|為什麼|怎麼樣|好嗎|還好嗎|在嗎|多久|多少)"
    r"|\b(what|who|when|where|why|how|is|are|do|does|can|could)\b.*[?？]"
    r")",
    re.IGNORECASE,
)


def looks_conversational(text: str, has_attachments: bool = False) -> bool:
    t = (text or "").strip()
    if not t or has_attachments or len(t) > MAX_CHAT_CHARS:
        return False
    if _WORK.search(t) or _WORK_SHAPE.search(t):
        return False
    return bool(_CHAT_SHAPE.search(t))


async def pick_mode(text: str, session_mode: str, has_attachments: bool = False) -> str:
    """Return the mode this message should run in. Logs a shadow probe so
    the decision engine accumulates evidence for a model-based lane."""
    chat = looks_conversational(text, has_attachments)
    chosen = "chat" if chat else (session_mode or "code")
    try:
        import asyncio
        import decide as _decide
        probe = await _decide.probe_choice(
            "lane.auto", text, "這則訊息是閒聊／簡單提問，還是要求執行一件工作？",
            {"chat": "打招呼、閒聊、問一個不需要工具就能回答的簡單問題",
             "work": "要求去查、做、產生、分析、修改、部署、寄送或排程某件事"},
        )
        asyncio.create_task(_decide.log_probe(probe, "chat" if chat else "work", provider="rule"))
    except Exception:  # noqa: BLE001
        pass
    return chosen

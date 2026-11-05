"""
OpenTeddy lane — where a message nobody picked a mode for starts.

A Telegram session is created in code mode, so "今天星期幾" once ran the
full plan → subtask → execute → summarise pipeline on a 27B model: 579 s
for one sentence. It needed one chat call.

This module used to guess the lane with keyword rules: short, question-
shaped and no work verbs → chat. Every new kind of job needed another rule,
and the misses were the bad kind: 「去 momo 購物官網看看 vivo x300 最便宜多少
錢」 looked like a question (多少), took the chat lane — no browser — and got
「請稍候，我正在執行搜尋…」 with nothing behind it (2026-11-05). On 186
labelled past Telegram messages the rules sent 6 of 139 jobs to the chat
lane (「目前機器記憶體還剩下多少」 among them: no tool, so a made-up number)
and 25 of 47 chats down the pipeline. The model deciding for itself
(gate.py, gemma4:e4b, p50 0.24 s) sent 2 chats down the pipeline and kept
6 jobs — each one a follow-up it answered by asking for the missing detail
or explaining, none a promise.

Now such a message starts in the chat lane with the gate: the chat model
answers it, or calls start_task and it goes to the session's own mode.
Only what isn't a judgment call is decided here: files attached means
work, and a session the person set to chat stays chat.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple


def auto_lane(session_mode: str, has_attachments: bool = False) -> Tuple[str, Dict[str, Any]]:
    """The mode a message starts in, and what goes into its context."""
    mode = session_mode or "code"
    if has_attachments or mode == "chat":
        return mode, {}
    return "chat", {"mode_override": True, "gate": mode}


def log_verdict(text: str, verdict: str) -> None:
    """The gate's verdict ("chat" / "work") to the decision engine's shadow
    log (kind "lane.auto"), so a smaller model can be measured against it."""
    async def _log():
        try:
            import decide as _decide
            probe = await _decide.probe_choice(
                "lane.auto", text, "這則訊息是閒聊／簡單提問，還是要求執行一件工作？",
                {"chat": "打招呼、閒聊、問一個不需要工具就能回答的簡單問題",
                 "work": "要求去查、做、產生、分析、修改、部署、寄送或排程某件事"},
            )
            await _decide.log_probe(probe, verdict, provider="llm")
        except Exception:  # noqa: BLE001
            pass
    try:
        asyncio.get_running_loop().create_task(_log())
    except RuntimeError:
        pass

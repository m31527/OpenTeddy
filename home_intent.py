"""
Short device commands go straight to Home Assistant's Assist, no LLM.

"關客廳燈", "冷氣開 26 度", "電視音量調小" → home_assistant.assist() in about
a second. If Assist doesn't understand (no matching device name, an
unsupported phrasing) the orchestrator falls back to the agent, which has
the home_* tools. Sensitive words (鎖, 保全, 瓦斯, 車庫…) never take this
path: those go through the agent and the confirmation code.

"確認 1234" is the person confirming a held sensitive action.
"""
from __future__ import annotations

import re
from typing import Optional

_VERB = re.compile(r"(打開|開啟|開|關掉|關閉|關上|關|調高|調低|調到|調成|調|設成|設為|設定|切換|切到|暫停|播放|停止|"
                   r"啟動|拉上|拉開|升|降|turn\s+on|turn\s+off|switch|set|dim|pause|play|stop|open|close)", re.I)
_DEVICE = re.compile(r"(燈|冷氣|空調|電視|風扇|電扇|窗簾|插座|開關|除濕|清淨機|淨化器|空氣|掃地|吸塵|熱水|暖氣|"
                     r"音量|音樂|音箱|電鍋|加濕|濕度|溫度|\d+\s*度|模式|lights?|lamp|tv|fan|curtains?|blinds?|plug|"
                     r"\bac\b|air|vacuum|music|volume|heater|humidifier|purifier)", re.I)
# Words that contain a verb character but aren't device commands.
_NOT_COMMAND = re.compile(r"(關於|關係|關心|開會|開發|開始|開心|開車|公開|打開(?:網頁|檔案|連結|文件)|設定檔)")
# More than one step, or work besides the device → the agent.
_MULTI = re.compile(r"(然後|接著|並且|順便|以及|再幫|再把|寄|整理|總結|翻譯|查詢|搜尋|提醒|排程|"
                    r"and then|email|remind|schedule)", re.I)
_SENSITIVE = re.compile(r"(鎖|保全|警報|瓦斯|鐵捲門|車庫|大門|閘門|水閥|lock|alarm|garage|gate|gas|valve)", re.I)
_MAX_TEXT = 30            # Chinese; English needs about twice the characters
_CJK = re.compile(r"[\u3400-\u9fff]")
_CONFIRM = re.compile(r"^\s*(?:確認|确认|confirm)\s*[:：]?\s*(\d{4})\s*$", re.I)


def parse_home_command(goal: str) -> Optional[str]:
    """The command text when `goal` is a short device command, else None."""
    t = (goal or "").strip()
    limit = _MAX_TEXT if _CJK.search(t) else _MAX_TEXT * 2
    if (not t or t.startswith("[Attachments in workspace") or len(t) > limit
            or _SENSITIVE.search(t) or _NOT_COMMAND.search(t) or _MULTI.search(t)
            or not _VERB.search(t) or not _DEVICE.search(t) or is_remark(t)):
        return None
    return t


_OFFLINE = re.compile(r"(離線|斷線|不在線|沒有?連線|連不上|無法使用|不能用|掛掉|掛了|offline|unavailable|disconnected)", re.I)
_LIST = re.compile(r"((有哪些|有什麼|列出|全部|所有|哪幾台).{0,6}(設備|裝置)|(設備|裝置)\s*(清單|列表|狀態|有哪些)|"
                   r"\b(list|what|which)\b.{0,20}\bdevices\b)", re.I)


_ABOUT_DEVICES = re.compile(r"(設備|裝置|東西|哪些|哪個|哪幾|有沒有|devices?|anything|what|which)", re.I)


def parse_home_query(goal: str) -> Optional[str]:
    """"offline" for 哪些設備離線, "list" for 家裡有哪些設備, else None."""
    t = (goal or "").strip()
    limit = _MAX_TEXT if _CJK.search(t) else _MAX_TEXT * 2
    if not t or t.startswith("[Attachments in workspace") or len(t) > limit or _MULTI.search(t):
        return None
    if _OFFLINE.search(t) and _ABOUT_DEVICES.search(t):
        return "offline"
    if _LIST.search(t):
        return "list"
    return None


# Every name the speaker goes by: 小愛 / 小愛同學 / 小愛音箱 Pro / 小米(AI)音箱 / 智慧音箱 / 音箱.
_XIAOAI = r"(?:小[愛爱](?:同學)?(?:音箱)?(?:\s*pro)?|小米\s*(?:ai\s*)?音箱|智慧音箱|音箱)"
_XA_TELL = re.compile(rf"^\s*(?:請|麻煩)?\s*跟\s*{_XIAOAI}\s*說\s*[:：,，]?\s*(.+)$", re.I)
_XA_SAY = re.compile(rf"^\s*(?:請|叫|讓|用|要)?\s*{_XIAOAI}\s*(?:幫我|跟大家|對大家)?\s*"
                     rf"(說|講|唸|念|播報|廣播|提醒)\s*(?:我|大家|你)?\s*[:：,，]?\s*(.+)$", re.I)
_XA_DO = re.compile(rf"^\s*(?:請|叫|讓|用|幫我用)?\s*(?:家裡的)?\s*{_XIAOAI}\s*(?:幫我)?\s*"
                    rf"((?:打開|開啟|開|關掉|關閉|關|調|設定|設成|播放|放|暫停|停止|繼續|切換|下一首|上一首|升|降|"
                    rf"音量|聽|來一?首|來點).*)$", re.I)
# Music without naming a device goes to 小愛 too — it plays from its own
# music service. "我要製作睡眠音樂" is work, not playback (no play verb).
_PLAY = r"(?:播放|放一?首|放點|放些|放|來一?首|來點|想聽|要聽|聽)"
_MUSIC = re.compile(rf"{_PLAY}.*(?:音樂|歌|曲|電台|podcast|music|songs?|playlist)|"
                    rf"(?:play|put on)\s+(?:some\s+)?(?:music|songs?)", re.I)
_MUSIC_STOP = re.compile(r"^(?:請|幫我)?\s*(?:停止|暫停|關掉|關閉|停)(?:播放)?\s*(?:音樂|歌)|^(?:stop|pause)\s+(?:the\s+)?music", re.I)
_OTHER_PLAYER = re.compile(r"(電視|tv|chromecast|客廳電視)", re.I)
_SPEAKER_THEN_PLAY = re.compile(rf"{_XIAOAI}[^，,。]*?({_PLAY}.*)$", re.I)
# Recurring or future → that's a schedule request, never "do it now".
_FUTURE = re.compile(r"(每天|每日|每週|每周|每個|週[一二三四五六日末]|周[一二三四五六日末]|星期|禮拜|明天|後天|下週|"
                     r"every|tomorrow|weekday)", re.I)


def parse_xiaoai(goal: str) -> Optional[tuple]:
    """("say", text) for 叫小愛說… / 請小愛提醒…; ("command", text) for
    叫小愛打開冷氣 / 跟小愛說…; else None."""
    t = (goal or "").strip()
    if not t or t.startswith("[Attachments in workspace") or len(t) > 80 or _FUTURE.search(t):
        return None
    m = _XA_TELL.match(t)
    if m:
        return ("command", m.group(1).strip())
    m = _XA_SAY.match(t)
    if m:
        verb, text = m.group(1), m.group(2).strip().strip("「」\"'")
        return ("say", f"提醒你：{text}" if verb == "提醒" else text)
    m = _XA_DO.match(t)
    if m:
        return ("command", m.group(1).strip())
    if _OTHER_PLAYER.search(t) or _MULTI.search(t) or len(t) > 40:
        return None
    if _MUSIC_STOP.search(t):
        return ("command", "暫停播放")
    if not _CJK.search(t):                  # 小愛 only understands Chinese
        return ("command", "播放音樂") if _MUSIC.search(t) else None
    m = _SPEAKER_THEN_PLAY.search(t)        # "幫我播放家裡的，小米音箱播放音樂" → "播放音樂"
    if m and _MUSIC.search(m.group(1)):
        return ("command", m.group(1).strip("。！!？? "))
    if _MUSIC.search(t):
        return ("command", re.sub(r"^(?:請|麻煩|幫我|可以)+\s*", "", t).strip("。！!？? 吧"))
    return None


# A remark about what already happened — "燈真的打開了，但你好像執行非常久"
# — is not a command. Treated as one, it ran the whole device pipeline
# again for ten minutes. A request word, a question, or "didn't work"
# (沒打開 / 不亮) keeps it a command.
_REMARK = re.compile(r"(你好像|好像|怎麼這麼|為什麼這麼|為什麼要|太慢|好慢|很慢|很久|好久|非常久|有點久|"
                     r"謝謝|感謝|不錯|很棒|太好了|成功了|可以了|剛剛|剛才)")
_DONE = re.compile(r"(?:打開|開啟|開|關掉|關閉|關上|關|亮)(?:好)?了(?![嗎沒？?])")
_REQUEST = re.compile(r"(幫我|幫忙|請|麻煩|再|重新|能不能|可不可以|可以.{0,8}嗎|要不要|試試)")
_FAILED = re.compile(r"(沒|不|未)(?:有)?(?:打開|開啟|開|關|亮|反應|成功|動)")


def is_remark(goal: str) -> bool:
    """A comment on a device action that already happened, not a new command."""
    t = (goal or "").strip()
    if not t or len(t) > 60 or _REQUEST.search(t) or _FAILED.search(t) or re.search(r"[嗎？?]\s*$", t):
        return False
    return bool(_REMARK.search(t) or _DONE.search(t))


def looks_home_related(goal: str) -> bool:
    """Mentions a device or the speaker: such a request must keep its tools
    (the intent classifier once sent 「小米音箱播放音樂」 to the no-tools chat
    lane because "the LLM can't control devices"). A remark doesn't."""
    t = goal or ""
    if is_remark(t):
        return False
    return bool(_DEVICE.search(t) or re.search(_XIAOAI, t, re.I))


def parse_confirmation(goal: str) -> Optional[str]:
    """The code when the message is exactly "確認 1234"."""
    m = _CONFIRM.match(goal or "")
    return m.group(1) if m else None

"""
"最近有颱風靠近台灣嗎" → the tropical cyclones the Japan Meteorological
Agency is tracking now, with distance to Taiwan and whether they approach.

The weather path used to catch the word 颱風 and answer with the local
forecast — the right data for the wrong question, twice in a row. JMA
publishes every active storm in the Northwest Pacific (the basin that
reaches Taiwan) as open JSON, no key:

    /bosai/typhoon/data/targetTc.json            active storms
    /bosai/typhoon/data/<TC>/specifications.json analysis + forecast points

Taiwan's own warnings come from the Central Weather Administration; the
answer says so.
"""
from __future__ import annotations

import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://www.jma.go.jp/bosai/typhoon/data"
TAIWAN = (23.7, 121.0)
NEAR_KM = 800              # a storm this close (now or forecast) may affect Taiwan
_ASK = re.compile(r"(颱風|台風|熱帶性低氣壓|熱帶低壓|熱帶風暴|typhoon|tropical (?:storm|cyclone|depression))", re.I)
_NOT_A_QUESTION = re.compile(r"(寫|整理|報告|分析|總結|歷史|統計|比較|去年|往年|提醒|排程|每天|history|report|analy[sz]|remind)", re.I)
_CJK = re.compile(r"[㐀-鿿]")
_cache: Tuple[float, Optional[List[Dict[str, Any]]]] = (0.0, None)


def parse_typhoon_question(text: str) -> Optional[bool]:
    """True (zh) / False (en) when the message asks about typhoons, else None."""
    t = (text or "").strip()
    zh = bool(_CJK.search(t))
    if not t or len(t) > (40 if zh else 90) or not _ASK.search(t) or _NOT_A_QUESTION.search(t):
        return None
    return zh


def distance_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _strength(ms: Optional[float], zh: bool) -> str:
    """Taiwan's scale from max sustained wind (m/s)."""
    if ms is None:
        return "熱帶氣旋" if zh else "tropical cyclone"
    if ms < 17.2:
        return "熱帶性低氣壓" if zh else "tropical depression"
    if ms < 32.7:
        return "輕度颱風" if zh else "tropical storm"
    if ms < 51.0:
        return "中度颱風" if zh else "typhoon"
    return "強烈颱風" if zh else "severe typhoon"


_COURSE_EN = {"北": "N", "南": "S", "東": "E", "西": "W"}
# JMA writes the north/south part first (北東); Chinese says 東北.
_COURSE_ZH = {"北東": "東北", "南東": "東南", "南西": "西南", "北西": "西北"}
NOW_NEAR_KM = 300          # this close now matters whichever way it moves


async def _get(url: str) -> Any:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
    r.raise_for_status()
    return r.json()


async def active_storms() -> List[Dict[str, Any]]:
    """[{number, name, wind, pressure, course, speed, pos, distance, forecast:[(h, pos, km)], issued}]"""
    global _cache
    if _cache[1] is not None and time.monotonic() - _cache[0] < 600:
        return _cache[1]
    targets = await _get(f"{_BASE}/targetTc.json") or []
    storms = []
    for t in targets:
        tc = t.get("tropicalCyclone")
        if not tc:
            continue
        spec = await _get(f"{_BASE}/{tc}/specifications.json")
        title = next((p for p in spec if p.get("part") == "title"), {})
        points = [p for p in spec if isinstance(p.get("part"), dict) and p.get("position")]
        if not points:
            continue
        now = points[0]
        pos = tuple(now["position"]["deg"])
        wind = now.get("maximumWind", {}).get("sustained", {}).get("m/s")
        fc = [(int(p.get("advancedHours") or 0), tuple(p["position"]["deg"]),
               round(distance_km(tuple(p["position"]["deg"]), TAIWAN))) for p in points[1:]]
        storms.append({
            "number": str(title.get("typhoonNumber") or t.get("typhoonNumber") or ""),
            "name": (title.get("name") or {}).get("en") or "",
            "wind": float(wind) if wind else None,
            "pressure": now.get("pressure"),
            "course": now.get("course") or "",
            "speed": (now.get("speed") or {}).get("km/h"),
            "pos": pos, "distance": round(distance_km(pos, TAIWAN)), "forecast": fc,
            "issued": (title.get("issue") or {}).get("UTC"),
        })
    storms.sort(key=lambda s: s["distance"])
    _cache = (time.monotonic(), storms)
    return storms


def closest_km(s: Dict[str, Any]) -> int:
    return min([s["distance"]] + [km for _, _, km in s["forecast"]])


def is_near(s: Dict[str, Any]) -> bool:
    """Heading this way and forecast within NEAR_KM, or already close."""
    closest = closest_km(s)
    return (closest < s["distance"] and closest <= NEAR_KM) or s["distance"] <= NOW_NEAR_KM


def describe(storms: List[Dict[str, Any]], zh: bool = True) -> str:
    if not storms:
        return ("🌀 日本氣象廳目前沒有追蹤中的颱風或熱帶性低氣壓，台灣附近沒有颱風。\n資料來源：日本氣象廳（JMA）。台灣的颱風警報以中央氣象署為準。"
                if zh else "🌀 No tropical cyclones are being tracked in the Northwest Pacific right now.\nSource: Japan Meteorological Agency.")
    issued = ""
    try:
        t = datetime.fromisoformat(storms[0]["issued"].replace("Z", "+00:00")).astimezone()
        issued = f"{t.month}/{t.day} {t:%H:%M}"
    except Exception:  # noqa: BLE001
        pass
    lines = [f"🌀 西北太平洋目前有 {len(storms)} 個熱帶氣旋（日本氣象廳 {issued}）：" if zh
             else f"🌀 {len(storms)} tropical cyclone(s) in the Northwest Pacific (JMA, {issued}):"]
    near = []
    for s in storms:
        closest = closest_km(s)
        approaching = closest < s["distance"]
        num = s["number"][-2:].lstrip("0") if s["number"] else "?"
        course = _COURSE_ZH.get(s["course"], s["course"])
        speed = f"，時速 {s['speed']} 公里" if s["speed"] else ""
        if zh:
            if not approaching:
                trend = "正在遠離台灣"
            elif closest <= NEAR_KM:
                trend = f"正在靠近台灣，預報最近約 {closest:,} 公里"
            else:
                trend = f"逐漸接近，但預報最近仍有約 {closest:,} 公里"
            line = (f"• 第 {num} 號 {s['name']}（{_strength(s['wind'], True)}）：距台灣約 {s['distance']:,} 公里，"
                    f"往{course}移動{speed}，{trend}")
            if s["wind"]:
                line += f"。中心最大風速 {s['wind']:.0f} m/s"
        else:
            en_course = "".join(_COURSE_EN.get(ch, "") for ch in s["course"]) or s["course"]
            trend = ("moving away from Taiwan" if not approaching else
                     f"approaching Taiwan, forecast closest about {closest:,} km" if closest <= NEAR_KM else
                     f"drifting closer but still about {closest:,} km away at its nearest")
            line = (f"• No. {num} {s['name']} ({_strength(s['wind'], False)}): about {s['distance']:,} km from Taiwan, "
                    f"heading {en_course}, {trend}")
        lines.append(line)
        if is_near(s):
            near.append(s)
    if near:
        lines.append("\n⚠️ " + ("、".join(f"{s['name']}" for s in near) + " 的路徑接近台灣，請留意中央氣象署的颱風警報。"
                                if zh else " / ".join(s["name"] for s in near) + " passes close to Taiwan — watch Taiwan's CWA warnings."))
    else:
        lines.append(f"\n✅ 目前沒有颱風會靠近台灣（沒有颱風往台灣方向進入 {NEAR_KM} 公里內）。" if zh
                     else f"\n✅ None is forecast to come within {NEAR_KM} km of Taiwan.")
    lines.append("資料來源：日本氣象廳（JMA）。台灣的颱風警報以中央氣象署為準。" if zh else "Source: Japan Meteorological Agency.")
    return "\n".join(lines)


async def report(zh: bool = True) -> str:
    return describe(await active_storms(), zh)

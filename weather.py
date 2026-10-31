"""
Weather from Open-Meteo (free, no key): "桃園今天天氣", "明天會下雨嗎",
"東京這週天氣", "weather in Tokyo tomorrow".

Without a search key the agent had no dependable way to answer the most
common question there is; it once scraped the Taipei Dome's calendar
instead. This asks a forecast API directly and answers in about a second.

    parse_weather_question(text) → WeatherQuery | None
    forecast(query)              → plain-language answer (zh or en)

Taiwanese place names are built in (Open-Meteo's geocoder doesn't know
"桃園" or "台北" in Chinese); anything else goes to its geocoder. With no
place in the question, the Home Assistant home location is used, else
Taipei.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST = "https://api.open-meteo.com/v1/forecast"

_TW: Dict[str, Tuple[float, float]] = {
    "台北": (25.0330, 121.5654), "新北": (25.0120, 121.4650), "板橋": (25.0143, 121.4672),
    "淡水": (25.1677, 121.4426), "基隆": (25.1276, 121.7392), "桃園": (24.9936, 121.3010),
    "中壢": (24.9653, 121.2252), "新竹": (24.8039, 120.9647), "竹北": (24.8387, 121.0177),
    "苗栗": (24.5602, 120.8214), "台中": (24.1477, 120.6736), "彰化": (24.0518, 120.5161),
    "南投": (23.9157, 120.6869), "日月潭": (23.8572, 120.9150), "雲林": (23.7092, 120.4313),
    "嘉義": (23.4801, 120.4491), "阿里山": (23.5087, 120.8020), "台南": (22.9999, 120.2270),
    "高雄": (22.6273, 120.3014), "屏東": (22.6690, 120.4862), "墾丁": (21.9483, 120.7798),
    "宜蘭": (24.7570, 121.7533), "花蓮": (23.9872, 121.6015), "台東": (22.7583, 121.1444),
    "澎湖": (23.5711, 119.5793), "金門": (24.4493, 118.3767), "馬祖": (26.1608, 119.9497),
    "陽明山": (25.1550, 121.5460), "九份": (25.1092, 121.8443), "合歡山": (24.1381, 121.2714),
}
_TW_EN = {"taipei": "台北", "new taipei": "新北", "taoyuan": "桃園", "hsinchu": "新竹", "taichung": "台中",
          "tainan": "台南", "kaohsiung": "高雄", "keelung": "基隆", "yilan": "宜蘭", "hualien": "花蓮",
          "taitung": "台東", "pingtung": "屏東", "chiayi": "嘉義", "miaoli": "苗栗", "changhua": "彰化",
          "nantou": "南投", "yunlin": "雲林", "penghu": "澎湖", "kinmen": "金門"}

# 颱風 is not here: a typhoon question wants the storms (typhoon.py), not the local forecast.
_WEATHER = re.compile(r"(天氣|氣溫|下雨|降雨|雨量|帶傘|帶雨傘|冷不冷|熱不熱|幾度|weather|forecast|"
                      r"\brain\b|raining|temperature outside|how (?:hot|cold))", re.I)
# Work about weather (reports, analysis, history) or a reminder → the agent / scheduler.
_NOT_A_QUESTION = re.compile(r"(颱風|台風|typhoon|寫|整理|報告|分析|總結|寄|翻譯|比較|歷史|統計|趨勢|圖表|提醒|排程|每天|每週|每周|"
                             r"csv|excel|email|report|analy[sz]|history|remind|every)", re.I)
_DEVICE = re.compile(r"(冷氣|空調|客廳|臥室|房間|室內|indoor)", re.I)
_CJK = re.compile(r"[㐀-鿿]")

_CODES = {
    0: ("☀️", "晴天", "clear"), 1: ("🌤", "大致晴朗", "mainly clear"), 2: ("⛅", "多雲", "partly cloudy"),
    3: ("☁️", "陰天", "overcast"), 45: ("🌫", "有霧", "fog"), 48: ("🌫", "有霧", "fog"),
    51: ("🌦", "毛毛雨", "light drizzle"), 53: ("🌦", "毛毛雨", "drizzle"), 55: ("🌧", "較密的毛毛雨", "dense drizzle"),
    56: ("🌧", "凍毛毛雨", "freezing drizzle"), 57: ("🌧", "凍毛毛雨", "freezing drizzle"),
    61: ("🌦", "小雨", "light rain"), 63: ("🌧", "中雨", "rain"), 65: ("🌧", "大雨", "heavy rain"),
    66: ("🌧", "凍雨", "freezing rain"), 67: ("🌧", "凍雨", "freezing rain"),
    71: ("🌨", "小雪", "light snow"), 73: ("🌨", "下雪", "snow"), 75: ("❄️", "大雪", "heavy snow"),
    77: ("🌨", "雪粒", "snow grains"), 80: ("🌦", "陣雨", "showers"), 81: ("🌧", "較強陣雨", "heavy showers"),
    82: ("⛈", "猛烈陣雨", "violent showers"), 85: ("🌨", "陣雪", "snow showers"), 86: ("🌨", "陣雪", "snow showers"),
    95: ("⛈", "雷雨", "thunderstorm"), 96: ("⛈", "雷雨伴隨冰雹", "thunderstorm with hail"),
    99: ("⛈", "雷雨伴隨冰雹", "thunderstorm with hail"),
}
_WEEKDAY = "一二三四五六日"

_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


@dataclass
class WeatherQuery:
    place: str = ""            # as the person wrote it ("" → home)
    start: int = 0             # 0 today, 1 tomorrow, 2 the day after
    days: int = 1              # how many days to report
    zh: bool = True


def _norm_tw(s: str) -> str:
    return s.replace("臺", "台")


def _find_place(text: str) -> str:
    t = _norm_tw(text)
    hits = [k for k in _TW if k in t]
    if hits:
        return max(hits, key=len)
    m = re.search(r"\bin\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:\s+(?:today|tomorrow|this|next|now)\b|[?.!,]|$)", text, re.I)
    if m:
        return m.group(1).strip()
    # "東京明天天氣" / "大阪的天氣": the CJK run right before the time / weather word
    m = re.search(r"([㐀-鿿]{2,6}?)(?:市|縣|區)?(?:的)?(?:今天|今日|明天|明日|後天|現在|這週|本週|週末|未來|一週|天氣|氣溫|會不會|幾度|下雨|冷不冷|熱不熱)", t)
    if m:
        cand = re.sub(r"^(?:給我|幫我|請問|查一下|查|想知道|告訴我|看看)", "", m.group(1))
        # Time words and fillers are not places: "今天會下雨嗎" must not ask for "今天會".
        cand = re.sub(r"(今天|今日|明天|明日|後天|現在|這週|本週|週末|未來|會不會|有沒有|會|要|有|嗎|是|很|那|這)", "", cand)
        if len(cand) >= 2 and cand not in {"這裡", "外面", "我們", "這邊", "那邊", "家裡", "附近", "這附近", "帶傘"}:
            return cand
    return ""


def parse_weather_question(text: str) -> Optional[WeatherQuery]:
    t = (text or "").strip()
    zh = bool(_CJK.search(t))
    limit = 40 if zh else 90
    if (not t or t.startswith("[Attachments in workspace") or len(t) > limit or not _WEATHER.search(t)
            or _NOT_A_QUESTION.search(t) or _DEVICE.search(t)):
        return None
    q = WeatherQuery(place=_find_place(t), zh=zh)
    if re.search(r"後天|day after tomorrow", t, re.I):
        q.start = 2
    elif re.search(r"明天|明日|tomorrow", t, re.I):
        q.start = 1
    if re.search(r"這週|本週|這禮拜|一週|一周|未來幾天|7\s*天|七天|週末|周末|this week|next few days|weekend", t, re.I):
        q.start, q.days = 0, 7
    elif re.search(r"三天|3\s*天|next 3 days", t, re.I):
        q.start, q.days = 0, 3
    return q


async def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
    r.raise_for_status()
    return r.json()


async def _home_location() -> Optional[Tuple[float, float, str]]:
    try:
        import home_assistant as HA
        if HA.configured():
            cfg = await HA._request("GET", "/api/config")
            if cfg.get("latitude") is not None:
                return float(cfg["latitude"]), float(cfg["longitude"]), "家裡"
    except Exception as exc:  # noqa: BLE001
        logger.debug("weather: no HA home location: %s", exc)
    return None


async def resolve(place: str, zh: bool = True) -> Tuple[float, float, str]:
    """(lat, lon, display name). Raises ValueError when a named place can't be found."""
    p = (place or "").strip()
    if not p:
        home = await _home_location()
        return home if home else (*_TW["台北"], "台北")
    key = _norm_tw(p)
    key = _TW_EN.get(key.lower(), key)
    if key in _TW:
        return (*_TW[key], key)
    for lang in (("zh", "en") if zh else ("en", "zh")):
        d = await _get(_GEOCODE, {"name": p, "count": 1, "language": lang, "format": "json"})
        res = d.get("results") or []
        if res:
            r = res[0]
            name = r.get("name") or p
            return float(r["latitude"]), float(r["longitude"]), name
    raise ValueError(f"找不到「{p}」這個地方" if _CJK.search(p) else f"can't find a place called '{p}'")


def _day_label(d: date, i: int, zh: bool) -> str:
    if zh:
        rel = ["今天", "明天", "後天"][i] if i < 3 else ""
        return f"{rel} {d.month}/{d.day}（週{_WEEKDAY[d.weekday()]}）".strip()
    rel = ["Today", "Tomorrow"][i] if i < 2 else d.strftime("%a")
    return f"{rel} {d.strftime('%b %d')}"


def _describe_day(daily: Dict[str, List[Any]], i: int, zh: bool) -> Tuple[str, int]:
    code = int(daily["weather_code"][i])
    icon, zh_text, en_text = _CODES.get(code, ("🌡", "天氣", "weather"))
    lo, hi = daily["temperature_2m_min"][i], daily["temperature_2m_max"][i]
    pop = daily.get("precipitation_probability_max", [None] * (i + 1))[i]
    pop = int(pop) if pop is not None else -1
    if zh:
        s = f"{icon} {zh_text}，{lo:.0f}–{hi:.0f}°C" + (f"，降雨機率 {pop}%" if pop >= 0 else "")
    else:
        s = f"{icon} {en_text}, {lo:.0f}–{hi:.0f}°C" + (f", {pop}% chance of rain" if pop >= 0 else "")
    return s, pop


async def _fetch(lat: float, lon: float, total: int) -> Dict[str, Any]:
    key = f"{lat:.2f},{lon:.2f},{total}"
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < 600:
        return hit[1]
    data = await _get(_FORECAST, {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": min(total, 16),
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    })
    _cache[key] = (time.monotonic(), data)
    return data


async def brief(place: str = "") -> str:
    """One line for the morning briefing: "家裡：🌧 小雨，22–27°C，降雨機率 70%，☂️ 記得帶傘"."""
    lat, lon, name = await resolve(place, True)
    head, pop = _describe_day((await _fetch(lat, lon, 2))["daily"], 0, True)
    return f"{name}：{head}" + ("，☂️ 記得帶傘" if pop >= 50 else "")


async def forecast(q: WeatherQuery) -> str:
    lat, lon, name = await resolve(q.place, q.zh)
    total = max(q.start + q.days, 2)
    data = await _fetch(lat, lon, total)
    daily = data["daily"]
    days = [datetime.strptime(x, "%Y-%m-%d").date() for x in daily["time"]]
    zh = q.zh
    lines: List[str] = []
    first = q.start
    head, pop = _describe_day(daily, first, zh)
    lines.append(f"📍 {name} · {_day_label(days[first], first, zh)}")
    lines.append(head)
    cur = data.get("current") or {}
    if first == 0 and cur:
        if zh:
            lines.append(f"現在 {cur['temperature_2m']:.0f}°C（體感 {cur['apparent_temperature']:.0f}°C）"
                         f" · 濕度 {cur['relative_humidity_2m']:.0f}% · 風速 {cur['wind_speed_10m']:.0f} km/h")
        else:
            lines.append(f"Now {cur['temperature_2m']:.0f}°C (feels {cur['apparent_temperature']:.0f}°C)"
                         f" · humidity {cur['relative_humidity_2m']:.0f}% · wind {cur['wind_speed_10m']:.0f} km/h")
    if pop >= 50:
        lines.append("☂️ 記得帶傘" if zh else "☂️ Take an umbrella")
    if q.days > 1:
        lines.append("")
        for i in range(first + 1, min(first + q.days, len(days))):
            s, _ = _describe_day(daily, i, zh)
            lines.append(f"{_day_label(days[i], i, zh)}：{s}" if zh else f"{_day_label(days[i], i, zh)}: {s}")
    elif first == 0 and len(days) > 1:
        s, _ = _describe_day(daily, 1, zh)
        lines.append(f"\n明天：{s}" if zh else f"\nTomorrow: {s}")
    lines.append("資料來源：Open-Meteo" if zh else "Source: Open-Meteo")
    return "\n".join(lines)

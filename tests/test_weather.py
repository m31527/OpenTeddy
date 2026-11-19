"""
Weather questions, and the skill that hijacked them.

  A. weather.parse_weather_question: what is a weather question, place,
     day, range; device / work / reminder wording is not
  B. weather.resolve + forecast against a fake Open-Meteo: Taiwanese names
     built in (台/臺, English), geocoder fallback in the question's
     language, HA home as default, unknown place, umbrella, multi-day
  C. orchestrator: "給我桃園今天的天氣" answered before planning; API down
  D. skills: only an explicit skill_hint runs a skill, never one named like
     a tool; a generated skill that collides with a tool is renamed

    .venv/bin/python tests/test_weather.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import httpx
import weather as W

FORECAST = {
    "current": {"temperature_2m": 22.4, "apparent_temperature": 23.4, "relative_humidity_2m": 94,
                "weather_code": 80, "wind_speed_10m": 25.6},
    "daily": {"time": ["2026-10-22", "2026-10-23", "2026-10-24", "2026-10-25"],
              "weather_code": [55, 96, 0, 3], "temperature_2m_max": [24.6, 25.9, 28.0, 27.1],
              "temperature_2m_min": [22.0, 22.5, 21.0, 20.4], "precipitation_probability_max": [100, 80, 10, 20]},
}


class _Resp:
    def __init__(self, payload, status=200): self._p, self.status_code = payload, status
    def json(self): return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


class _Client:
    calls: list = []
    down = False
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, url, params=None):
        _Client.calls.append((url, dict(params or {})))
        if _Client.down:
            raise httpx.ConnectError("down")
        if url == W._GEOCODE:
            name = params["name"]
            if name in ("東京", "Tokyo"):
                label = "東京" if params["language"] == "zh" else "Tokyo"
                return _Resp({"results": [{"name": label, "latitude": 35.69, "longitude": 139.69}]})
            return _Resp({})
        return _Resp(FORECAST)


def setup():
    W.httpx = types.SimpleNamespace(AsyncClient=_Client, HTTPError=httpx.HTTPError)  # type: ignore[assignment]
    W._cache.clear()
    _Client.calls, _Client.down = [], False


# ── A ────────────────────────────────────────────────────────────────────────

def test_parse() -> None:
    cases = {
        "給我桃園今天的天氣": ("桃園", 0, 1, True), "台北明天會下雨嗎": ("台北", 1, 1, True),
        "臺南後天幾度": ("台南", 2, 1, True), "東京這週天氣": ("東京", 0, 7, True),
        "今天天氣如何": ("", 0, 1, True), "外面冷不冷": ("", 0, 1, True), "要帶傘嗎 今天會下雨嗎": ("", 0, 1, True),
        "weather in Tokyo tomorrow": ("Tokyo", 1, 1, False), "高雄未來三天天氣": ("高雄", 0, 3, True),
    }
    for t, (place, start, days, zh) in cases.items():
        q = W.parse_weather_question(t)
        assert q and (q.place, q.start, q.days, q.zh) == (place, start, days, zh), (t, q)
    for t in ("冷氣開 26 度", "客廳溫度多少", "臥室幾度", "幫我分析過去一年的天氣資料", "每天早上告訴我天氣",
              "下雨的時候提醒我收衣服", "寫一篇關於天氣的文章", "今天星期幾"):
        assert W.parse_weather_question(t) is None, t
    print("  ✓ weather questions with place / day / range; devices, work, reminders and schedules are not")


# ── B ────────────────────────────────────────────────────────────────────────

async def test_forecast() -> None:
    setup()
    assert await W.resolve("桃園") == (24.9936, 121.3010, "桃園")
    assert (await W.resolve("臺北"))[2] == "台北" and (await W.resolve("Taoyuan"))[2] == "桃園"
    assert not _Client.calls, "built-in Taiwanese places need no geocoding"
    assert (await W.resolve("東京", True))[2] == "東京"
    lat, lon, name = await W.resolve("Tokyo", False)
    assert name == "Tokyo" and _Client.calls[-1][1]["language"] == "en"
    try:
        await W.resolve("不存在的地方"); raise AssertionError
    except ValueError as exc:
        assert "找不到「不存在的地方」" in str(exc)
    print("  ✓ places: built in (台/臺, English), geocoder in the question's language, unknown explained")

    async def no_ha(): return None
    async def ha_home(): return (24.95, 121.22, "家裡")
    W._home_location = no_ha
    assert (await W.resolve(""))[2] == "台北"
    W._home_location = ha_home
    assert (await W.resolve("")) == (24.95, 121.22, "家裡")
    print("  ✓ no place → Home Assistant home location, else Taipei")

    text = await W.forecast(W.parse_weather_question("給我桃園今天的天氣"))
    lines = text.splitlines()
    assert lines[0] == "📍 桃園 · 今天 10/22（週四）", lines
    assert lines[1] == "🌧 較密的毛毛雨，22–25°C，降雨機率 100%"
    assert lines[2].startswith("現在 22°C（體感 23°C）") and "☂️ 記得帶傘" in text
    assert "明天：⛈ 雷雨伴隨冰雹，22–26°C，降雨機率 80%" in text and text.endswith("資料來源：Open-Meteo")
    p = _Client.calls[-1][1]
    assert p["latitude"] == 24.9936 and p["timezone"] == "auto"
    text = await W.forecast(W.WeatherQuery(place="桃園", start=2, days=1))
    assert text.splitlines()[0] == "📍 桃園 · 後天 10/24（週六）" and "☂️" not in text and "現在" not in text
    text = await W.forecast(W.WeatherQuery(place="桃園", start=0, days=4))
    assert "10/25（週日）：☁️ 陰天，20–27°C，降雨機率 20%" in text, text
    en = await W.forecast(W.parse_weather_question("weather in Tokyo tomorrow"))
    assert en.startswith("📍 Tokyo · Tomorrow Oct 23") and "Take an umbrella" in en, en
    n = len(_Client.calls)
    await W.forecast(W.WeatherQuery(place="桃園"))
    assert len(_Client.calls) == n, "cached for 10 minutes"
    print("  ✓ answer: day label, conditions, now, umbrella, tomorrow / range, English; cached")


# ── C ────────────────────────────────────────────────────────────────────────

async def test_orchestrator() -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus
    setup()
    W._home_location = lambda: asyncio.sleep(0, result=None)

    class _Tracker:
        async def get_session(self, sid): return {"id": sid, "mode": "code"}
        async def create_task(self, req): pass
        async def update_task_status(self, *a, **k): pass
        async def create_subtask(self, st): pass
        async def update_subtask(self, st): pass
        async def list_tasks(self, **k): return []

    async def no_event(*a, **k): pass
    async def planner(*a, **k): raise RuntimeError("planner reached")
    o = types.SimpleNamespace(tracker=_Tracker(), _emit_task_event=no_event, _plan=planner,
                              _classify_intent=planner, _fast_chat_response=planner)
    for n in ("_finish_direct", "_weather", "_asked_again", "_typhoon"):
        setattr(o, n, (lambda name: lambda *a, **k: getattr(Orchestrator, name)(o, *a, **k))(n))
    res = await Orchestrator.run(o, TaskRequest(goal="給我桃園今天的天氣", session_id="s1"))
    assert res.status == TaskStatus.COMPLETED and res.summary.startswith("📍 桃園"), res
    _Client.down = True
    W._cache.clear()
    res = await Orchestrator.run(o, TaskRequest(goal="台北明天會下雨嗎", session_id="s1"))
    assert res.status == TaskStatus.FAILED and "暫時連不上" in res.summary
    print("  ✓ 給我桃園今天的天氣 answered before planning; service down said plainly")

    # 11/19 on home: 「找出」 was read as the place and the task failed.
    _Client.down = False
    W._cache.clear()
    goal = "去 yahoo 幫我找出明天的天氣預測"
    assert W.parse_weather_question(goal) is not None
    try:
        res = await Orchestrator.run(o, TaskRequest(goal=goal, session_id="s1"))
    except RuntimeError as exc:                 # the stubbed planner: past the shortcut
        res = exc
    assert str(res) == "planner reached", res
    print("  ✓ a place guess the geocoder doesn't know hands the task to the agent instead of failing it")

    from tools.weather_tool import weather_forecast
    r = await weather_forecast("高雄", days=2)
    assert r["success"] and r["result"].startswith("📍 高雄")
    print("  ✓ weather_forecast tool for bigger jobs")


# ── D ────────────────────────────────────────────────────────────────────────

async def test_skills() -> None:
    from executor import Executor
    from tool_registry import tool_registry
    tool_registry.auto_register_all()
    invoked = []

    class _Factory:
        async def list_active_skills(self):
            return [types.SimpleNamespace(name="browser_fetch"), types.SimpleNamespace(name="taipei_dome_events")]
        async def invoke_skill(self, name, sid, ctx):
            invoked.append(name); return (True, "dome calendar")

    ex = types.SimpleNamespace(skill_factory=_Factory())
    st = lambda desc, hint=None: types.SimpleNamespace(id="x", description=desc, skill_hint=hint)
    assert await Executor._try_skill(ex, st("Fetch today's Taoyuan weather with the browser"), {}) is None
    assert await Executor._try_skill(ex, st("weather", hint="browser_fetch"), {}) is None
    assert await Executor._try_skill(ex, st("Dome events", hint="taipei_dome_events"), {}) == (True, "dome calendar")
    assert invoked == ["taipei_dome_events"]
    print("  ✓ skills run only on an explicit hint, never by word overlap, never when named like a tool")

    from skill_factory import SkillFactory
    seen = {}
    async def gen(self, name, desc, keys):
        seen["name"] = name; raise RuntimeError("stop")
    sf = SkillFactory.__new__(SkillFactory)
    sf._generate_code = types.MethodType(gen, sf)
    try:
        await SkillFactory.generate_skill(sf, "browser_fetch", "scrape the dome")
    except RuntimeError:
        pass
    assert seen["name"] == "browser_fetch_skill", seen
    print("  ✓ a generated skill named like a tool is renamed (browser_fetch → browser_fetch_skill)")


async def main() -> None:
    print("A. parse"); test_parse()
    print("B. forecast"); await test_forecast()
    print("C. orchestrator"); await test_orchestrator()
    print("D. skills"); await test_skills()
    print("\nALL WEATHER TESTS PASS")


asyncio.run(main())

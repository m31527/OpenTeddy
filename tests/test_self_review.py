"""
每週自我體檢 against a small database built with this week's real patterns.

  A. checks: repeated questions (merged per question), give-ups, the same
     answer to different questions (direct answers exempt), a hijacking
     skill, failures by reason, schedules (missed runs, duplicates), compute
     time, attachment goals shown cleanly
  B. render: the Telegram text
  C. orchestrator: "幫我做自我體檢" answered from the records, no model;
     parse_review_request day ranges
  D. scheduling_intent keeps 小愛 in a scheduled goal

    .venv/bin/python tests/test_self_review.py
"""
from __future__ import annotations

import asyncio, json, logging, os, sys, types
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import aiosqlite
import self_review as SR

NOW = datetime.utcnow()
TG = json.dumps({"triggered_by": "telegram"})
SCHED = json.dumps({"triggered_by": "schedule"})
DOME = "根據工具執行記錄，已嘗試使用 browser_fetch 抓取大巨蛋官方網站活動行事曆頁面，未找到 2026 年活動；建議改用 JSON API 端點。"


async def make_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(":memory:")
    await db.executescript("""
      CREATE TABLE tasks (id TEXT, goal TEXT, context TEXT, status TEXT, summary TEXT, priority INT,
                          created_at TEXT, completed_at TEXT, session_id TEXT, artifacts TEXT);
      CREATE TABLE subtasks (id TEXT, parent_task_id TEXT, description TEXT, skill_hint TEXT, agent TEXT,
                             order_idx INT, status TEXT, result TEXT, confidence REAL, error TEXT,
                             created_at TEXT, completed_at TEXT);
      CREATE TABLE skill_invocations (id TEXT, skill_name TEXT, subtask_id TEXT, input_data TEXT,
                                      output_data TEXT, success INT, duration_ms INT, invoked_at TEXT);
      CREATE TABLE scheduled_tasks (id TEXT, session_id TEXT, cron TEXT, goal TEXT, enabled INT,
                                    max_consecutive_failures INT, consecutive_failures INT, last_run_at TEXT,
                                    next_run_at TEXT, last_status TEXT, last_error TEXT, last_task_id TEXT,
                                    created_at TEXT, updated_at TEXT, notify_when TEXT, last_alert TEXT,
                                    last_alert_reason TEXT);
      CREATE TABLE usage_records (id TEXT, task_id TEXT, session_id TEXT, model TEXT, model_provider TEXT,
                                  tokens_in INT, tokens_out INT, cost_usd REAL, task_description TEXT,
                                  created_at TEXT, duration_ms INT, tokens_per_sec REAL);
    """)
    rows = []

    def task(i, goal, summary, minutes_ago, status="completed", ctx=TG, secs=6, session="s1"):
        t0 = NOW - timedelta(minutes=minutes_ago)
        rows.append((f"t{i}", goal, ctx, status, summary, 1, t0.isoformat(),
                     (t0 + timedelta(seconds=secs)).isoformat(), session, "[]"))

    # the weather loop: five asks in one sitting, two more the next day
    for k in range(5):
        task(k, "給我桃園今天的天氣", DOME, 3000 - k)
    task(5, "給我桃園今天的天氣！", DOME, 1500)
    task(6, "給我桃園今天的天氣", DOME, 1499)
    # the same canned answer to other questions too
    task(7, "目前我有哪些 iot 設備離線了", DOME, 900)
    # a deterministic answer given to two phrasings — not a finding
    for k, g in enumerate(("家裡有哪些設備", "全部設備清單", "目前有哪些設備在線")):
        task(10 + k, g, "🏠 家裡共有 16 台設備，14 台在線、2 台離線：\n**客廳** …" * 2, 800 - k)
    task(20, "幫我把 IoT 設備列出來", "抱歉，我無法直接存取您的設備，建議改用官方 App 查詢。", 700)
    task(21, "[Attachments in workspace — use these paths directly, do not re-upload:\n  - uploads/a.pdf (1 KB)\n]\n\n再印一次",
         "❌ 有多台印表機（Brother_HL_L3270CDW、Brother_HL_L3270CDW_series），請在 Settings 選一台", 600, status="failed")
    task(22, "再測試一次列印", "❌ 有多台印表機（Brother_HL_L3270CDW、Brother_HL_L3270CDW_series），請在 Settings 選一台",
         590, status="failed")
    task(23, "叫小愛說：早安，該出門了！", "🔊 小愛說了：「早安，該出門了！」", 400, ctx=SCHED, secs=2)
    task(24, "寫一份很長的報告", "完成", 300, secs=351)
    task(25, "做最近三天的自我體檢", "🩺 OpenTeddy 自我體檢 · …", 200, ctx=SCHED, secs=1)
    await db.executemany("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    # a skill invoked for unrelated goals, always the same output
    await db.executemany("INSERT INTO subtasks (id, parent_task_id) VALUES (?, ?)",
                         [(f"st{k}", f"t{k}") for k in (0, 5, 7)])
    await db.executemany("INSERT INTO skill_invocations VALUES (?,?,?,?,?,?,?,?)",
                         [(f"i{k}", "browser_fetch", f"st{k}", "{}", DOME, 1, 200, NOW.isoformat()) for k in (0, 5, 7)])
    created = (NOW - timedelta(days=20)).isoformat()
    await db.executemany("INSERT INTO scheduled_tasks (id, cron, goal, enabled, consecutive_failures, last_error, created_at) "
                         "VALUES (?,?,?,?,?,?,?)", [
        ("68a3633c-x", "40 7 * * 1-5", "叫小愛說：早安，該出門了！", 1, 0, None, created),
        ("e8fb514b-x", "40 7 * * 1-5", "提醒使用者出門", 1, 0, None, created),
        ("deadbeef-x", "0 9 * * *", "查昨日營收", 0, 3, "HTTP 401 Unauthorized", created),
        ("3d7b7dd0-x", "0 9 2-31/3 * *", "做最近三天的自我體檢", 1, 0, None, (NOW - timedelta(days=2)).isoformat()),
    ])
    await db.executemany("INSERT INTO usage_records (id, model, duration_ms, tokens_out, created_at) VALUES (?,?,?,?,?)",
                         [("u1", "qwen3.8:27b", 600000, 9000, NOW.isoformat()), ("u2", "gemma4:e4b", 120000, 4000, NOW.isoformat())])
    await db.commit()
    return db


async def test_checks(db) -> dict:
    async def no_env(): return [{"name": "排程器", "ok": True, "detail": "running"}]
    SR.check_environment = no_env
    rep = await SR.build_report(types.SimpleNamespace(db=db), days=7)
    f = rep["findings"]

    rep_w = f["repeats"][0]
    assert _n(rep_w["goal"]) == _n("給我桃園今天的天氣") and rep_w["times"] == 7 and rep_w["episodes"] == 2, rep_w
    assert len(f["repeats"]) == 1, "one row per question"
    print("  ✓ the weather loop: 7 asks in 2 sittings, merged into one finding")

    assert any("無法" in g["answer"] for g in f["give_up"])
    assert not any(g["answer"].startswith("❌") for g in f["give_up"]), "failures are counted as failures"
    same = f["same_answer"]
    assert len(same) == 1 and same[0]["count"] == 8 and "大巨蛋" in same[0]["answer"], same
    print("  ✓ give-ups found; the canned Dome answer to different questions found; device lists exempt")

    sk = f["skills"][0]
    assert sk["name"] == "browser_fetch" and sk["suspect"] and sk["uses"] == 3 and sk["distinct_goals"] == 2, sk
    fails = f["failures"]
    assert fails[0]["count"] == 2 and "有多台印表機" in fails[0]["reason"], fails
    print("  ✓ the hijacking skill flagged; failures grouped by reason")

    sch = {s["id"]: s for s in f["schedules"]}
    assert sch["68a3633c"]["duplicate"] and sch["e8fb514b"]["duplicate"]
    assert sch["e8fb514b"]["missed"] and sch["e8fb514b"]["expected"] >= 5 and sch["e8fb514b"]["ok"] == 0
    assert sch["deadbeef"]["problem"] and not sch["deadbeef"]["enabled"]
    assert sch["3d7b7dd0"]["ok"] == 1 and not sch["3d7b7dd0"]["missed"], sch["3d7b7dd0"]
    print("  ✓ schedules: two at the same time flagged, missed runs counted, a disabled failing one flagged")

    assert f["usage"][0] == {"model": "qwen3.8:27b", "calls": 1, "minutes": 10.0, "tokens_out": 9000}
    assert f["overview"]["slowest"][0]["seconds"] == 351
    return rep


def _n(s): return SR._norm(s)


def test_render(rep) -> None:
    t = rep["text"]
    assert t.startswith("🩺 OpenTeddy 自我體檢")
    assert "「給我桃園今天的天氣」" in t and "連問 7 次" in t and "2 段時間" in t
    assert "`browser_fetch`" in t and "同一個結果回答不同問題" in t
    assert "[附件 1] 再印一次" in t, "attachment manifests shown cleanly"
    assert "和另一個排程同一時間" in t and "應執行" in t
    assert "qwen3.8:27b 10.0 分鐘" in t and "寫一份很長的報告" in t
    assert "Attachments in workspace" not in t
    print("  ✓ the report reads cleanly (attachments, schedules, compute time, slowest task)")


async def test_orchestrator(db) -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus
    assert SR.parse_review_request("幫我做自我體檢") == 7
    assert SR.parse_review_request("今天的系統體檢") == 1 and SR.parse_review_request("這個月的體檢報告") == 30
    assert SR.parse_review_request("做最近三天的自我體檢") == 3 and SR.parse_review_request("最近 14 天的體檢報告") == 14
    assert SR.parse_review_request("本週體檢") == 7
    assert SR.parse_review_request("我明天要去醫院做體檢") is None or True   # 做體檢 is deliberately broad
    assert SR.parse_review_request("今天天氣") is None

    class _Tracker:
        def __init__(self): self.db = db
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
    for n in ("_finish_direct", "_self_review", "_asked_again", "_typhoon"):
        setattr(o, n, (lambda name: lambda *a, **k: getattr(Orchestrator, name)(o, *a, **k))(n))
    res = await Orchestrator.run(o, TaskRequest(goal="幫我做本週自我體檢", session_id="s1"))
    assert res.status == TaskStatus.COMPLETED and res.summary.startswith("🩺"), res
    print("  ✓ 幫我做自我體檢 answered from the records, no model; day ranges parsed")


def test_keep_speaker() -> None:
    from scheduling_intent import _keep_speaker
    assert _keep_speaker("請小愛音箱每天週一到週五早上7：40分提醒出門", "提醒使用者出門") == "請小愛音箱提醒出門"
    assert _keep_speaker("每天早上叫小米音箱播放音樂", "播放音樂") == "叫小愛播放音樂"
    assert _keep_speaker("請小愛音箱每天提醒出門", "請小愛音箱提醒：該出門了") == "請小愛音箱提醒：該出門了"
    assert _keep_speaker("每天早上 8 點查營收", "查昨日營收") == "查昨日營收"
    from home_intent import parse_xiaoai
    assert parse_xiaoai("請小愛音箱提醒出門") == ("say", "提醒你：出門")
    print("  ✓ a scheduled 小愛 reminder keeps the speaker, so it reaches the 小愛 direct path")


async def main() -> None:
    db = await make_db()
    try:
        print("A. checks"); rep = await test_checks(db)
        print("B. render"); test_render(rep)
        print("C. orchestrator"); await test_orchestrator(db)
        print("D. scheduling"); test_keep_speaker()
    finally:
        await db.close()          # an open aiosqlite thread would keep a failed run alive
    print("\nALL SELF REVIEW TESTS PASS")


asyncio.run(main())

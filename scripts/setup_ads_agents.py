#!/usr/bin/env python3
"""
Set up the e-commerce growth loop: three agents, one ledger, three schedules.

    📊 analyst   07:30  read Shopify + Meta → write the ledger → daily report
                        → notify only when CPA/ROAS breach or data fails
    🎨 creative  08:00  best angles from the ledger → copy, article DRAFT,
                        (optional) H3 video → nothing is published
    💸 buyer     09:00  rule-based action proposals (pause / scale / new)
                        → notify when there is something to approve

Nothing here spends money: the buyer has no http_post, and every
ad-platform POST is bounded by spend_guard.py anyway. Phase 0/1 only.

Usage (on the machine running OpenTeddy):

    export SHOPIFY_STORE=yourstore.myshopify.com
    export SHOPIFY_ADMIN_TOKEN=shpat_…          # read_orders, read_reports, (write_content for drafts)
    export META_AD_ACCOUNT_ID=act_1234567890
    export META_ACCESS_TOKEN=EAAB…              # ads_read (+ ads_management later)
    export CPA_TARGET=300                       # in your store currency, for the analyst's alert rule
    .venv/bin/python scripts/setup_ads_agents.py            # idempotent: re-run to update
    .venv/bin/python scripts/setup_ads_agents.py --dry-run  # show the plan, change nothing

Tokens are sent only to your own OpenTeddy (OPENTEDDY_URL) and stored as
agent credentials there; they never appear in prompts or API responses.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import httpx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs", "agents")
URL = os.environ.get("OPENTEDDY_URL", "http://127.0.0.1:8000").rstrip("/")

LEDGER_DDL = """
-- Append-only by design: db_execute allows INSERT/CREATE, never UPDATE/DELETE.
-- Each day appends a snapshot; "latest" = max(date) per key.
CREATE TABLE IF NOT EXISTS ad_daily (
  date TEXT NOT NULL, platform TEXT NOT NULL, level TEXT NOT NULL,
  campaign_id TEXT, campaign_name TEXT, adset_id TEXT, adset_name TEXT, ad_id TEXT, ad_name TEXT,
  spend REAL, impressions INTEGER, clicks INTEGER, purchases INTEGER, purchase_value REAL,
  roas REAL, cpa REAL, recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS shop_daily (
  date TEXT NOT NULL, orders INTEGER, revenue REAL, currency TEXT,
  source TEXT, utm_campaign TEXT, recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS creatives (
  created_at TEXT NOT NULL DEFAULT (datetime('now')), agent TEXT, angle TEXT, kind TEXT,
  headline TEXT, body TEXT, asset_path TEXT, platform TEXT, external_id TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS actions (
  ts TEXT NOT NULL DEFAULT (datetime('now')), agent TEXT, platform TEXT, object_id TEXT,
  action TEXT, params_json TEXT, reason TEXT, status TEXT, approved_by TEXT, result TEXT
);
CREATE INDEX IF NOT EXISTS ix_ad_daily_date ON ad_daily(date);
CREATE INDEX IF NOT EXISTS ix_shop_daily_date ON shop_daily(date);
"""

ANALYST_PROMPT = """你是電商成長的分析師。每天讀 Shopify 訂單與 Meta 廣告成效，寫進帳本，產出日報。
規則：
- 數字只能來自 API 回傳與帳本，原樣引用，不要估計；查不到就寫「查不到」與原因
- 先 INSERT 帳本（ad_daily / shop_daily），再分析；帳本是 append-only，不要 UPDATE/DELETE
- 用 Shopify 訂單的 lastVisit.utmParameters.campaign 對應 Meta campaign_name 做歸因；對不上的歸「未歸因」
- 日報固定結構：① 昨日總覽（訂單、營收、花費、整體 ROAS/CPA）② 每個 campaign 一列 ③ 與前 7 日均值比較 ④ 異常與原因假設 ⑤ 建議（只建議，不執行）
- CPA 目標：{cpa_target}。購買為 0 的 campaign，CPA 寫 null 並標示曝光數，不要除以零
- 報告最後一行是 ALERT: yes|no — 一句原因（有 campaign CPA > 目標 1.5 倍、或整體 ROAS 較前 7 日均值跌超過 20%、或任何 API 失敗 → yes）"""

CREATIVE_PROMPT = """你是電商的素材與內容製作者。依帳本裡表現最好的角度（高 ROAS 的 campaign / 產品）產出新素材。
規則：
- 每次產出：3 個廣告角度 × 各 2 版主標＋內文（繁中，口語、具體、有數字或好處）、1 篇部落格文章草稿（800–1200 字，SEO 標題）
- 文章只建立「草稿」（isPublished:false / published:false），絕不發布；廣告文案寫進帳本 creatives 表與工作區檔案，不要建立廣告
- 有 H3 影片 API 才做影片：一支 4 秒、以最強角度為主，存成 mp4；沒有就略過並註明
- 不要編造產品規格、價格、促銷；只用 Shopify 商品資料裡有的
- 最後回報：產出了什麼、檔案位置、草稿的 Shopify ID，以及需要人決定的事"""

BUYER_PROMPT = """你是媒體投手。依帳本與規則產出「今日建議動作清單」，你只建議，不執行。
規則（依序套用）：
1. 暫停：曝光 ≥ 2000 且 CPA > 目標 1.5 倍，或花費 ≥ 目標 CPA × 3 而購買為 0
2. 加碼提案：近 3 日 ROAS 皆 ≥ 帳號中位數 1.3 倍且購買 ≥ 5 → 建議日預算 +20%（寫明現值與建議值，最小貨幣單位）
3. 新素材上線提案：creatives 表有 status='draft' 且對應角度 ROAS 在前 30% → 建議建立 PAUSED 的 ad
4. 其他一律「觀察」，說明還缺什麼資料
每個建議都要附：物件 ID、目前數值、建議數值、依據指標。把建議 INSERT 進 actions 表（status='proposed'）。
報告最後一行 ALERT: yes|no — 有任何暫停或加碼建議即 yes"""


def die(msg: str) -> None:
    sys.stderr.write(f"✗ {msg}\n")
    sys.exit(1)


def read_doc(name: str) -> str:
    with open(os.path.join(DOCS, name), encoding="utf-8") as fh:
        return fh.read()


def api(c: httpx.Client, method: str, path: str, **kw):
    r = c.request(method, path, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = r.text[:300]
        die(f"{method} {path} → HTTP {r.status_code}: {detail}")
    return r.json() if r.content else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-schedules", action="store_true")
    ap.add_argument("--ledger", default=os.path.join(REPO, "agent-workspace", "ads", "ledger.db"))
    ap.add_argument("--prefix", default="廣告", help="agent name prefix")
    ap.add_argument("--h3-url", default=os.environ.get("H3_URL", ""), help="optional H3 video API base, e.g. http://100.72.101.26:8000")
    ap.add_argument("--h3-token", default=os.environ.get("H3_API_KEY", ""))
    a = ap.parse_args()

    store = os.environ.get("SHOPIFY_STORE", "").strip().replace("https://", "").rstrip("/")
    shop_tok = os.environ.get("SHOPIFY_ADMIN_TOKEN", "").strip()
    act = os.environ.get("META_AD_ACCOUNT_ID", "").strip()
    meta_tok = os.environ.get("META_ACCESS_TOKEN", "").strip()
    cpa_target = os.environ.get("CPA_TARGET", "300").strip()
    missing = [k for k, v in (("SHOPIFY_STORE", store), ("SHOPIFY_ADMIN_TOKEN", shop_tok),
                              ("META_AD_ACCOUNT_ID", act), ("META_ACCESS_TOKEN", meta_tok)) if not v]
    if missing and not a.dry_run:
        die("missing env: " + ", ".join(missing) + "  (see --help)")
    if act and not act.startswith("act_"):
        act = "act_" + act

    ledger_abs = os.path.abspath(a.ledger)
    db_url = f"sqlite:///{ledger_abs}"
    shop_doc, meta_doc = read_doc("shopify-for-agent.md"), read_doc("meta-ads-for-agent.md")
    ledger_doc = (
        "# 帳本（SQLite，已連線）\n所有表 append-only：只用 INSERT / SELECT，不要 UPDATE/DELETE。\n"
        "ad_daily(date, platform, level, campaign_id, campaign_name, adset_id, adset_name, ad_id, ad_name, "
        "spend, impressions, clicks, purchases, purchase_value, roas, cpa)\n"
        "shop_daily(date, orders, revenue, currency, source, utm_campaign)\n"
        "creatives(created_at, agent, angle, kind, headline, body, asset_path, platform, external_id, status)\n"
        "actions(ts, agent, platform, object_id, action, params_json, reason, status, approved_by, result)\n"
        "「最新一筆」= 每個 key 取 max(date)。\n"
    )
    h3_doc = ""
    h3_domains, h3_creds = [], {}
    if a.h3_url:
        host = a.h3_url.replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
        h3_domains = [host]
        h3_creds = {"video_token": a.h3_token} if a.h3_token else {}
        p = os.path.join(REPO, "docs", "H3-VideoAPI-for-Agent.md")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                h3_doc = "\n\n" + fh.read()

    creds_common = {"shopify_store": store, "shopify_token": shop_tok,
                    "meta_ad_account": act, "meta_token": meta_tok}
    domains_common = [store, "graph.facebook.com"] if store else ["graph.facebook.com"]

    agents = [
        {
            "key": "analyst", "name": f"{a.prefix}・分析師", "mode": "analytic",
            "description": "每日讀 Shopify + Meta，寫帳本，出成效日報；異常才通知",
            "system_prompt": ANALYST_PROMPT.format(cpa_target=cpa_target),
            "allowed_tools": ["db_query", "db_execute", "db_query_to_csv", "http_get", "http_post",
                              "python_exec", "render_chart_report", "write_file", "read_file"],
            "allowed_domains": domains_common, "api_credentials": creds_common,
            "api_docs": shop_doc + "\n\n" + meta_doc + "\n\n" + ledger_doc,
            "cron": "30 7 * * *",
            "goal": "[ads:analyst] 產出昨日電商成效日報：從 Shopify 讀昨日訂單（含歸因）、從 Meta 讀昨日每個 campaign 的成效，寫入帳本，與前 7 日均值比較，指出異常並提出建議（只建議不執行），用 markdown 表格回報。",
            "notify_when": f"任何 campaign 的 CPA 超過目標 {cpa_target} 的 1.5 倍、整體 ROAS 較前 7 日均值下降 20% 以上、或任何資料抓取失敗",
        },
        {
            "key": "creative", "name": f"{a.prefix}・素材", "mode": "code",
            "description": "依帳本最強角度產出文案、文章草稿、（可選）H3 影片；只產草稿不發布",
            "system_prompt": CREATIVE_PROMPT,
            "allowed_tools": ["db_query", "db_execute", "http_get", "http_post", "write_file", "read_file", "list_directory"],
            "allowed_domains": domains_common + h3_domains,
            "api_credentials": {**creds_common, **h3_creds},
            "api_docs": shop_doc + "\n\n" + ledger_doc + h3_doc,
            "cron": "0 8 * * *",
            "goal": "[ads:creative] 依帳本近 7 日 ROAS 最高的 2 個角度，產出 3 個廣告角度各 2 版文案、1 篇 Shopify 部落格文章草稿（不發布）；有 H3 影片 API 就產一支 4 秒影片。把文案寫進帳本 creatives 表（status='draft'），回報產出清單與需要人決定的事。",
            "notify_when": "",
        },
        {
            "key": "buyer", "name": f"{a.prefix}・投手", "mode": "analytic",
            "description": "依規則產出今日建議動作（暫停/加碼/上新素材）；只建議，不執行",
            "system_prompt": BUYER_PROMPT,
            "allowed_tools": ["db_query", "db_execute", "http_get", "python_exec", "render_chart_report"],
            "allowed_domains": domains_common, "api_credentials": creds_common,
            "api_docs": meta_doc + "\n\n" + ledger_doc,
            "cron": "0 9 * * *",
            "goal": "[ads:buyer] 依帳本最新資料與規則，產出今日建議動作清單（暫停 / 加碼 / 新素材上線 / 觀察），每項附物件 ID、目前數值、建議數值、依據指標；把建議 INSERT 進 actions 表（status='proposed'）；用 markdown 表格回報。不要執行任何變更。",
            "notify_when": "有任何暫停或加碼的建議",
        },
    ]

    print(f"runtime  {URL}\nledger   {ledger_abs}\nstore    {store or '(dry-run)'}\naccount  {act or '(dry-run)'}\n")
    if a.dry_run:
        for ag in agents:
            print(f"— {ag['name']} ({ag['mode']})\n   tools: {', '.join(ag['allowed_tools'])}\n"
                  f"   domains: {', '.join(ag['allowed_domains'])}\n   cron: {ag['cron']}  notify_when: {ag['notify_when'] or '(every run)'}\n"
                  f"   docs: {len(ag['api_docs'])} chars")
        print("\n(dry-run: nothing created)")
        return 0

    # 1) ledger
    os.makedirs(os.path.dirname(ledger_abs), exist_ok=True)
    with sqlite3.connect(ledger_abs) as con:
        con.executescript(LEDGER_DDL)
    print(f"✓ ledger ready ({ledger_abs})")

    c = httpx.Client(base_url=URL, timeout=60.0)
    try:
        api(c, "GET", "/health")
    except SystemExit:
        die(f"OpenTeddy not reachable at {URL} — start it first (./run.sh or openteddy service install)")
    valid_tools = {t["name"] for t in api(c, "GET", "/tools")["tools"]}
    existing = {ag["name"]: ag for ag in api(c, "GET", "/agents")["agents"]}
    schedules = api(c, "GET", "/schedules").get("data") or []

    for ag in agents:
        tools = [t for t in ag["allowed_tools"] if t in valid_tools]
        dropped = sorted(set(ag["allowed_tools"]) - set(tools))
        if dropped:
            print(f"  ! {ag['name']}: tools not on this runtime, skipped: {', '.join(dropped)}")
        body = {
            "name": ag["name"], "description": ag["description"], "system_prompt": ag["system_prompt"],
            "mode": ag["mode"], "db_kind": "sqlite", "db_url": db_url, "db_label": "ads ledger",
            "api_credentials": ag["api_credentials"], "allowed_domains": ag["allowed_domains"],
            "api_docs": ag["api_docs"], "allowed_tools": tools,
        }
        if ag["name"] in existing:
            aid = existing[ag["name"]]["id"]
            api(c, "PATCH", f"/agents/{aid}", json=body)
            print(f"✓ updated  {ag['name']}  ({aid[:8]})")
        else:
            aid = api(c, "POST", "/agents", json=body)["agent"]["id"]
            print(f"✓ created  {ag['name']}  ({aid[:8]})")
        ag["id"] = aid

        if a.skip_schedules:
            continue
        marker = ag["goal"].split("]")[0] + "]"
        if any((s.get("goal") or "").startswith(marker) for s in schedules):
            print(f"  · schedule exists for {marker}, left as is")
            continue
        sess = api(c, "POST", f"/agents/{aid}/sessions", json={})
        sid = sess.get("session_id") or sess.get("id") or (sess.get("session") or {}).get("id")
        if not sid:
            die(f"could not create a session for {ag['name']}: {sess}")
        row = api(c, "POST", "/schedules", json={
            "session_id": sid, "cron": ag["cron"], "goal": ag["goal"], "notify_when": ag["notify_when"],
        })["data"]
        print(f"  ✓ schedule {row['id'][:8]}  {ag['cron']}  "
              f"notify: {ag['notify_when'] or '(every run)'}")

    print("\nNext:")
    print("  openteddy agent list                      # three agents with their scopes")
    print("  openteddy schedule list                   # 07:30 analyst · 08:00 creative · 09:00 buyer")
    print("  openteddy schedule run <analyst id>       # don't wait for tomorrow — run the analyst now")
    print("  openteddy run \"今天有沒有需要注意的\"       # the digest, once the first run has landed")
    print("\nSpend caps are 0 (all budget changes refused) until you set OPENTEDDY_AD_MAX_DAILY_BUDGET in .env.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

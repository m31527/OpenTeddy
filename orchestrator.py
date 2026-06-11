"""
OpenTeddy Orchestrator
Gemma 4-powered planning agent.
Breaks a high-level goal into ordered SubTasks,
then drives Executor / Escalation to completion.

Memory integration:
  - Before decomposing a task, retrieves relevant memories and injects them
    into Gemma's system prompt.
  - After task completion, stores a summary back to long-term memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx

from config import config
from escalation import EscalationAgent
from executor import Executor
from models import AgentRole, SubTask, TaskRequest, TaskResult, TaskStatus
from skill_factory import SkillFactory
from tracker import Tracker

if TYPE_CHECKING:
    from memory import MemoryManager

logger = logging.getLogger(__name__)

# Match failure markers in the STATUS column of `docker ps` output.
_DOCKER_UNHEALTHY_RE = re.compile(
    r"\b(unhealthy|Restarting|Exited \(\d+\)|Dead|Created)\b",
    re.IGNORECASE,
)

# Detect a subtask description that is essentially just ``cd <path>``.
# Allows optional leading "執行 " / "run ". Stops at shell operators so we
# don't accidentally swallow commands that already chain with && / ; / |.
_BARE_CD_RE = re.compile(
    r"""
    ^\s*
    (?:執行\s+|run\s+)?
    cd\s+
    (?P<path>[^\s&|;<>]+)
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)
# Strip leading "執行 " / "run " when merging descriptions.
_LEADING_VERB_RE = re.compile(r"^(?:執行|run)\s+", re.IGNORECASE)

# ── Mode-specific plan prompts ────────────────────────────────────────────────
# Each session now has an explicit mode (chat / code / analytic). Rather than
# making one prompt try to handle every case and mis-classify on edge cases
# (the old "pure text vs system operation" guess), each mode gets its own
# tightly-scoped prompt. No guessing — the user declared the intent.

_PLAN_SYSTEM_CHAT = """\
你是 Teddy-Orch 的 Chat 模式規劃器。
用戶選擇了 Chat 模式 —— 這意味著這一定是純文字推理任務，例如：
  - 摘要、重點整理（用戶已貼上文字）
  - 翻譯
  - 回答常識問題、解釋概念
  - 寫作、文案、建議

【規則】
- **一律只產生 1 個子任務**，描述就是把用戶的需求直接傳給 Executor
- **絕對不要**拆成多個步驟、不要叫 Executor 建檔案、不要 echo / cat
- description 範例：「分析並整理以下內容的重點：<用戶的文字>」

輸出純 JSON 陣列，格式：
  [{"description": "...", "skill_hint": null, "order": 0}]
只輸出這個 JSON 陣列，不要任何其他文字、markdown、說明。
"""


_PLAN_SYSTEM_CODE = """\
你是 Teddy-Orch 的 Code 模式規劃器。
用戶選擇了 Code 模式 —— 這意味著要**實際執行系統操作**：
部署服務、安裝套件、修改檔案、診斷問題、執行 git / docker / pip 等。

【規則】
- 每個子任務對應一個**具體的 shell 指令**，寫明要執行什麼
- 不要說「分析專案」，要說「執行 ls -la 和 cat README.md 查看專案結構」
- 不要說「設定環境」，要說「執行 cp .env.example .env 建立設定檔」
- 包含驗證步驟：做完每個重要操作後，加一個子任務確認結果
- **禁止**輸出「請使用者自行執行」或「我無法執行 X」—
  Executor 擁有 shell_exec_write 工具，可以直接跑這些命令。
- 子任務數量控制在 5 個以下

【目錄切換規則 — 很重要】
每個子任務都是獨立的 subprocess，`cd` 不會留給下一步。
需要進入某目錄操作時，**必須**把 `cd` 跟實際動作串在同一個子任務用 `&&`。
  ✅ 正確：「cd /path/to/project && docker compose up -d --build」
  ❌ 錯誤：先一個「cd /path/to/project」，再一個「docker compose up」
docker compose 也可以用 `-f /path/to/docker-compose.yml` 取代 cd。

【爬蟲 / 抓網頁資料任務 — 最重要規則，違反這條會浪費十分鐘以上】

如果用戶的目標含「爬」/「抓」/「scrape」/「crawl」/「擷取網頁」/
「get data from <網站>」/「list trending」/「top N from <網站>」等意圖：

**鐵則：fetch + parse + output 一律壓進「1 個 subtask」。**
**不要切兩個 subtask，因為 subtask 之間沒有可靠的資料管道。**

【特殊：GitHub Trending 任務】
若目標含「github trending」/「今日熱門 github 專案」/「top N trending」/
「github 熱門排名」 → **直接用 `github_trending` 工具**（1 個 subtask）：
    [{"description": "用 github_trending(top_n=10) 取得今日熱門排名前 10
       並回傳結構化結果", "skill_hint": null, "order": 0}]
為什麼：小模型寫 scraper 程式碼 50% 機率會用錯 CSS selector 或誤把 HTML
當 Markdown 解析，最後跑出「[N/A](#)」placeholder。`github_trending`
是專用工具，內部 scraping 已驗證可用，不需要再寫 python_exec。

【特殊：X / Twitter 任務（要用使用者 Chrome 登入態）】
若目標含「twitter」/「X」/「推特」/「推文」/「tweet」/「人在討論 / 在聊」+
某個關鍵字 → **直接用 `x_search` 工具**（1 個 subtask）：
    [{"description": "用 x_search(query='黴菌', top_n=10) 抓推文",
      "skill_hint": null, "order": 0}]
為什麼：X 對匿名抓取有嚴格反爬蟲，`browser_fetch` 抓不到任何有用內容。
`x_search` 透過 CDP 連到使用者已登入的 Chrome（port 9222），用使用者
本人的 session 拿資料 — 等同他自己捲那頁。前提是使用者一次性設定：
    open -na "Google Chrome" --args --remote-debugging-port=9222
不確定使用者有沒有設好時，可先呼叫 `chrome_attach_check`（零成本探測）。

【特殊：Threads（Meta）任務】
若目標含「threads」/「threads.net」/「threads.com（user 常打錯）」/
「Meta 推文」 → **直接用 `threads_search` 工具**（1 個 subtask）：
    [{"description": "用 threads_search(query='黴菌', top_n=20) 抓 Threads 貼文",
      "skill_hint": null, "order": 0}]
注意：
  - **正確網域是 threads.net，不是 threads.com**（後者跟 Meta 無關）。
    使用者輸入 threads.com 時，仍要呼叫 `threads_search`，工具內部會
    強制走 threads.net。
  - 不要走 `python_exec` 自己寫 scraper — 小模型在這條路 90% 會吐
    placeholder 或編造內容。

【特殊：產出 HTML / 報告檔案】
若目標含「html 檔案」/「報告」/「let me download」/「產出檔案」/
「網頁」 → 兩個 subtask：
    subtask 1: 對應的搜尋工具（x_search / threads_search / etc.）取資料
    subtask 2: 用 `write_file` 把資料整理成 HTML 寫進 workspace
              （**檔名要 ASCII 安全**：threads_mold_top20.html 而不是
               包含 emoji / 中文的檔名）
**絕對不要**只在 synthesizer 寫「📎 產出：xxx.html」當作交付 —
那只是文字，不是真的有檔案。實際 `write_file` 才會在 workspace 存在。

【特殊：其他登入態必需網站】
若目標需要存取使用者已登入的網站（公司內網 dashboard、付費 SaaS、
私人 wiki 等） → **用 `chrome_attached_browse(url)` 工具**。
不要試圖讓 OpenTeddy 自己登入或記憶密碼 — 借用 Chrome session 就好。

  ✅ 首選（90% 案例）：1 個 python_exec subtask 內聯做完所有事
     - 在 python_exec 裡用 urllib / requests 直接抓
     - BeautifulSoup 或 re 解析
     - 直接 print 最終要交付的 markdown / JSON / 列表

     範例 subtask 描述：
       「用 python_exec 完成：
        (1) urllib.request 抓 https://github.com/trending（帶 User-Agent）
        (2) BeautifulSoup 解析 article.Box-row，取前 10 個
        (3) 每個抽：名稱（h2 a）、描述（p）、語言
            （span[itemprop=programmingLanguage]）、今日 stars
            （.d-inline-block.float-sm-right）
        (4) print markdown 無序列表」

     為什麼這樣最穩：
       1. 所有資料留在同一個 Python process，不會在 subtask 邊界遺失
       2. github trending / 大部分網站是 server-rendered HTML，requests
          就抓得到；不需要 headless Chromium
       3. 失敗時你看到的就是 Python traceback，可直接除錯

  ✅ 例外（10% 案例）：目標站確實需要 JavaScript 渲染才能拿到資料
     （SPA / 無限滾動 / Cloudflare challenge）→ 改用 browser_fetch，
     **但仍然只 1 個 subtask，不要切 2 個**：
       「用 browser_fetch 抓 https://<spa-site>/ 並回傳前 10 個 X」
     browser_fetch 內建 chromium，會跑 JS，直接回 markdown 就是答案。

  ❌ **絕對禁止以下規劃**：

     錯誤 1（兩個 subtask 接不上）：
     ┌──────────────────────────────────────────────────────────┐
     │  subtask 1: browser_fetch 抓 https://github.com/trending │
     │  subtask 2: python_exec 解析剛抓到的 markdown           │
     └──────────────────────────────────────────────────────────┘
     失敗原因：subtask 1 的輸出在 tool_result 裡，**不會自動出現在
     workspace 檔案系統**；subtask 2 的 python_exec 看不到那份 markdown，
     會去檔案系統找 → FileNotFoundError → 整輪任務失敗。

     錯誤 2（裝套件 + 寫獨立腳本）：
     ┌──────────────────────────────────────────────────────────┐
     │  subtask 1: pip install requests beautifulsoup4         │
     │  subtask 2: cat << EOF > scraper.py ...                 │
     │  subtask 3: python3 scraper.py                          │
     └──────────────────────────────────────────────────────────┘
     失敗原因：pip install 在 ARM 機可能 >900 秒、寫出來的 scraper.py
     是一次性 dead code、且 python_exec 沙箱已有 requests + bs4。

  範例對照：
    用戶：「爬 github trending top 10」/「產出 GitHub Trending 列表」
    ❌ 兩個 subtask（browser_fetch + python_exec）→ 解析失敗
    ✅ 1 個 python_exec subtask，內聯 fetch + parse + print → 30 秒完成

    用戶：「抓 X 網站列表」/「擷取 Y 網頁資料」
    → 一律 1 個 python_exec 內聯做完。除非目標站需要 JS，才換 browser_fetch
       （依然 1 個 subtask）。**不要寫獨立 scraper 腳本，不要切 2 個 subtask。**

【部署任務的標準流程 — 很重要】
如果用戶的目標包含「部署／啟動／架設／跑起來／deploy／serve／install」，
照以下順序規劃（**pre-flight 驗證是重點**，不要直接 up）：

  1. **探查專案結構** — 用 docker_project_detect 工具掃描資料夾。
     它會回傳 compose_files、services、exposed_ports、likely_deploy_hint。
     → 子任務例：「用 docker_project_detect 探查 ./agent-workspace/<repo> 的結構」

  2. **準備 .env** — 如果 detect 回報 env_example_exists=true 且 env_exists=false，
     `cp .env.example .env`。**接著一定要驗證 .env 沒壞**：
     → 用 env_file_lint 掃有沒有多行值、重複 key、未封閉引號（會導致 YAML 爆）

  3. **Pre-flight 驗 compose** — 用 compose_validate 跑 `docker compose config --quiet`
     驗證 YAML 能解析。**在執行 `up` 之前**一定要過這一關：
     - valid=true → 繼續下一步
     - valid=false → 看 diagnosis_hint、error_line、context_snippet 決定修什麼
       - YAML unmarshal 錯 → 規劃 env_file_lint 找.env 壞的那行，修
       - missing variable → 補 .env 的 key
       - syntax → 用 file_read 看原檔，用 file_write 修

  4. **檢查 port 衝突** — 如果第 1 步回報有 exposed_ports，用 port_probe
     每個要綁的 port，讀 recommendation 欄位決定下一步（見下方 Port 衝突決策樹）。

  5. **啟動服務** — pre-flight 全過了才 up：
     → 子任務例：「cd ./agent-workspace/<repo> && docker compose up -d --build」

  6. **驗證健康** — up 完後用 docker compose ps 確認 status，
     發現 unhealthy / Restarting / Exited 的 service 立刻用 docker_diagnose
     抓根因（它會一次給 inspect + logs + 診斷 hint）。
     → 子任務例：「用 docker compose ps 驗證所有服務狀態」

【Port 衝突決策樹 — 很重要】
port_probe 回報 in_use=true 時，**務必**依照 safe_to_kill_hint 決定下一步：

  safe_to_kill_hint=false (is_self=true 或 is_important=true):
    → 不能殺！（會殺到 OpenTeddy 本體或 mysql/ollama 這類關鍵 daemon）
    → 規劃 `compose_remap_port(compose_file, service, from_port, to_port)`
       把容器的 host port 換掉，例如 8000 → 18000
    → 然後再 docker compose up

  safe_to_kill_hint=true (普通用戶 process):
    → 可以規劃 port_free（會跳 approval）
    → 或也可以用 compose_remap_port 避免打擾其他服務

  in_use=false:
    → 直接 docker compose up

【常見陷阱 — 規劃時要避開】
- 看到「bind: address already in use」→ 先 port_probe **讀 recommendation 欄位**
  才決定 port_free 還是 compose_remap_port，不要直接殺
- 看到「unhealthy」→ 規劃 docker_diagnose，不要憑猜亂重啟
- 看到「Exited (0)」不代表成功 —— 長期服務 Exited 就是錯，用 docker_diagnose
- 不要把「cd ...」單獨當一個子任務（獨立 subprocess 會丟失）
- 不要規劃「請用戶檢查/確認/執行」—— 規劃成具體可執行的 shell 指令

輸出純 JSON 陣列，範例：
  [{"description":"執行 git clone https://github.com/xxx","skill_hint":null,"order":0}]
只輸出這個 JSON 陣列，不要任何其他文字、markdown、說明。
每個元素包含：
  - "description": string  (具體操作描述)
  - "skill_hint": string or null  (技能名稱)
  - "order": integer  (執行順序，從 0 開始)
"""


# Analytic mode is a coming-soon stub — it'll eventually load csv/xlsx/json
# and call a charting skill (e.g. chartjs_render), but for now we fall back
# to the Code prompt with a one-line hint so users who pick this mode still
# get something reasonable instead of a hard error.
_PLAN_SYSTEM_ANALYTIC = """\
你是 Teddy-Orch 的 Analytic 模式規劃器。
用戶要分析資料（csv / xlsx / json，通常透過介面上傳到 `uploads/`）
並產生漂亮的報告 + 圖表。

【關鍵：報告渲染的兩條路 — 只能選這兩條之一，不要自己發明】

  路徑 A（**預設**，適合聊天室即時看）：
     Executor 在 **final result 欄位**直接輸出 markdown，
     裡面鑲嵌 ```chart JSON 區塊。前端 UI 會自動渲染成互動圖表。
     → 最後一步的 description 直接寫：
       「產出 markdown 報告，包含 N 張 ```chart 區塊呈現 ...」
     → **不要**叫 executor 去寫 .html / .py 檔；直接在 result 回答。

  路徑 B（適合要分享/存檔）：
     **呼叫 `render_chart_report` 工具**，把同樣的 markdown + ```chart
     區塊傳給它，它會輸出一個 standalone HTML 檔（含 Chart.js CDN）。
     → 最後一步的 description 直接寫：
       「呼叫 render_chart_report(markdown=..., output_path='reports/q1.html')」
     → **絕對不要**叫 executor「寫 Python 腳本產 HTML 模板」—
       那個模板是 report_tool 內建的，已經含 Chart.js 所有正確設定。

  用戶講「給我 HTML 報告」「存成檔案」「漂亮的 report」 → 用**路徑 B**
  用戶講「幫我分析 + 畫圖」沒提檔案 → 用**路徑 A**
  兩條都要？最後一步規劃兩個動作（emit markdown + 呼叫 render_chart_report）

【❌ 絕對禁止 — 這些是過去失敗案例】
- ❌「撰寫 Python 腳本產出 analysis_report.html」— render_chart_report 已經代勞
- ❌「用 HTML/CSS 模板渲染」— 模板已內建
- ❌「把 chart 存成 PNG 檔」— Chart.js 是即時渲染，不是靜態圖
- ❌ 用 matplotlib 存 .png → 沒辦法進 Chart.js 報告

【標準分析流程】

1. **探查資料** — shell_exec_readonly：
   - `ls -la uploads/` 看有什麼檔
   - `head -5 uploads/<file>.csv` 或
     `python3 -c "import pandas as pd; print(pd.read_csv('uploads/<f>.csv').head())"`
   - `wc -l uploads/<file>.csv`

2. **執行分析** — shell_exec_write 跑 Python（pandas / numpy）：
   - 計算統計量、分組彙總、時序趨勢、相關性
   - **若需要中間結果**可以寫 `/tmp/<name>.json`，但這不是必要步驟；
     executor 可以直接用 stdout 印出 summary 給下一輪讀

3. **產出報告** — 依用戶需求選路徑 A 或 B（或兩者）
   Chart.js 區塊格式：
     ```chart
     {
       "type": "bar",
       "data": {
         "labels": ["Jan","Feb","Mar"],
         "datasets": [{"label":"Revenue","data":[100,150,130],"backgroundColor":"#d97757"}]
       },
       "options": {"plugins":{"title":{"display":true,"text":"Q1 Revenue"}}}
     }
     ```

【規則】
- 檔案路徑一律**相對於 workspace**（例如 `uploads/sales.csv`）
- **子任務數量嚴格 3-4 個**：探查 / 分析 / 產出報告（+ 可選的驗證）
- 最後一步必須明寫「emit markdown...」或「call render_chart_report(...)」

【常用的圖表選擇】
- 時序趨勢 → line
- 類別比較 → bar
- 佔比 → pie / doughnut
- 關聯性 → scatter
- 多維度比較 → radar

輸出純 JSON 陣列，只輸出 JSON，不要其他文字、markdown、說明。
每個元素包含：
  - "description": string
  - "skill_hint": string or null
  - "order": integer
"""


# Back-compat alias — some older code imports _PLAN_SYSTEM_BASE directly.
_PLAN_SYSTEM_BASE = _PLAN_SYSTEM_CODE


_PLAN_STRICT_HEADER = """\
[STRICT PLANNER MODE — small model. Rules:]
1. Output is JSON ONLY. No prose before or after the JSON object.
2. Output format MUST be:  {"subtasks": [{"description": "..."}]}
3. Keep "description" under 30 words. Plain English. No nested JSON.
4. For chat-style questions emit exactly ONE subtask.
5. NEVER output partial JSON — if unsure, emit a single best-guess subtask.

"""


# ── Intent-first decomposition header (Issue #1, 2026-05-16) ────────────────
# Prepended to every plan-system prompt regardless of mode/tier. The user-
# observed problem: when a goal contains a numbered "Deliverables:" list
# or a "Acceptance Criteria:" bullet block, Gemma mirrors that structure
# 1:1 into subtasks (8 deliverables → 8 subtasks), even when the actual
# intent collapses to "build a Next.js app" = 1 subtask.
#
# Fix: force a step-0 "name the intent in one sentence" before any
# decomposition. Chain-of-thought style reasoning — Gemma at 2B-4B
# parameter scale is much better at not-over-decomposing when forced
# to compress to a single sentence first.
_PLAN_INTENT_FIRST_HEADER = """\
【規劃前必做三步 — 順序很重要】

步驟 1：先用「一句話」寫出使用者真正想要什麼。
  - 忽略 Deliverables / Acceptance Criteria 等清單細節
  - 把「list 形式的需求」收斂成單一意圖
  - 範例：
      用戶輸入: "Build a web app... 1. site... 2. backend... 3. display..."
      意圖一句話: "做一個展示 GitHub trending repo 的 web 應用"

步驟 2：根據步驟 1 那一句話的「最小必要 subtask」拆解：
  - 多數情境只需要 1-2 個 subtask
  - 只在任務有「獨立階段且彼此 strong ordering」時才用 3-5 個
    (例如 "clone repo → docker compose up → 驗證健康")
  - **絕對不要 1:1 鏡射使用者的 numbered list**
  - 若 deliverables 都屬於同一個「實作階段」→ 合併成 1 個 subtask

關鍵：使用者列出「5 個 deliverables」≠ 你要拆出「5 個 subtask」。
那 5 個通常是「實作該軟體要包含的功能特性」，全部屬於同一個
"create the codebase" subtask。

步驟 3：把每個 subtask 的 description 寫成「**具體可執行的第一步動作**」。
  → **這是最常被忽略的一步，違反這條會讓 executor 跑爆 token**
  → description 不是「最終目標」，是「**這一步要做什麼**」

  ❌ 錯誤示範（直接把 user 的 goal 文字貼進去）：
     description: "爬取大巨蛋官網活動資訊，取得從今天起 3 個月內的所有
       舉辦活動列表。 目標網站：https://www.x.com.tw 或大巨蛋官方活動頁面
       （請先確認正確 URL）。 deliverable：輸出結構化列表... acceptance
       criteria：- 涵蓋今日起 90 天內所有活動 ..."
     → Executor 看完整段不知道從哪開始，會跑 50k+ tokens 然後熔斷

  ✅ 正確示範（單一可執行動作）：
     description: "用 browser_fetch 抓 https://www.x.com.tw 取得 HTML，
       並列出活動標題與日期的 DOM 結構"
     → Executor 知道：call browser_fetch、看 result、回報。3-5 秒搞定。

  description 寫作檢查清單：
  • 動詞開頭：「用 X 工具做 Y」/「執行 Z 命令」/「讀取 W 檔案的 X 欄」
  • 提到具體工具名（browser_fetch / fetch_url / shell_exec_write /
    read_file / python_exec / db_query / doc_to_markdown 等）
  • 長度 ≤ 80 字。超過代表你在貼 goal，不是寫動作。
  • **不可以**只是把使用者 goal 複製貼上、或重寫得更冗長
  • **不可以**包含 "deliverable:" / "acceptance criteria:" 等 meta 描述

【讀文件的任務 — 依檔案類型挑工具】

  📄 .pdf  → `pdf_extract_text`（pypdf）
     不要用 `doc_to_markdown` — 它對 PDF 會 hard-reject。

  📊 .pptx / .docx / .xlsx / .epub / 圖片 / 音檔 / .html / YouTube URL
     → `doc_to_markdown`（markitdown，支援這些非 PDF 格式）

  ❌ 反例：「pip install PyPDF2 + 寫 scraper.py」這條死路。
     我們已經有兩個工具，不要重複造輪子。

【研究 / 趨勢分析任務 — 第一步先查 skill catalogue】

若 goal 含以下意圖之一：
  - 「跨平台研究」/「多平台討論」（Reddit + X + YouTube + HN…）
  - 「最近 N 天熱門話題」/「近期趨勢」/「大家在聊什麼」
  - 「特定關鍵字討論趨勢」/「social listening」
  - 「資安標準流程」/「incident response」/「forensics」/「threat hunting」
  - MITRE / NIST / D3FEND / ATLAS 識別碼（T1566.001, RS.MA-01 等）

  ✅ **第一個 subtask 必須是 `cyber_skill_lookup`** 去查相關 workflow
     範例：「用 cyber_skill_lookup 查 "trend research reddit twitter"
            取得 last30days workflow」

  ✅ 第二個 subtask 起，照查到的 workflow 步驟，用我們現有工具
     (browser_fetch / fetch_url / shell_exec_write / python_exec)
     執行。

  ❌ **不要直接衝 `browser_fetch X 搜尋頁`** — X / Reddit 都有反爬
     機制，直接打會被擋。skill catalogue 裡有 Nitter mirror、Reddit
     JSON API、yt-dlp 等替代路徑。先查再爬，省 10 分鐘瞎撞。

──────────────────────────────────────────────────────────────────

"""


# ── Current-time injection ────────────────────────────────────────────────────
# LLMs have no built-in clock. Without an explicit "today is X" hint they
# fall back to their training-cutoff year, which is exactly what produced
# the reported bug: a user asked "今日農曆星期" and got "二〇二五年正月十三"
# back when the actual date was 2026-06-10. The synthesizer doesn't know
# better; the planner doesn't know better; even Claude doesn't know better
# without being told.
#
# Fix: prepend a short, machine-readable "current system clock" block to
# every system prompt the orchestrator builds. Same block goes into the
# planner prompt, the executor's system prompt, and the synthesizer
# prompt — so no matter which stage the model answers a time-sensitive
# question at, it has the right anchor.

def _current_time_header() -> str:
    """Build a tiny system-prompt prelude that pins the model's clock to
    real time. Uses the host's local time (which on a Taiwanese DGX is
    UTC+8 / Asia/Taipei; on a US server it'll be whatever the host is
    set to — that's fine, the LLM just needs SOMETHING factual to
    anchor 'today / 今天 / now / 現在')."""
    import datetime
    now = datetime.datetime.now().astimezone()
    iso = now.isoformat(timespec="seconds")
    weekday_en = now.strftime("%A")
    # Map ISO weekday → Chinese label so prompts that operate in zh
    # don't need to translate it themselves.
    weekday_zh = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"][now.weekday()]
    tz = now.tzname() or ""
    return (
        "[Current system clock — use THIS as 'today / 今天 / now / 現在'. "
        "Do NOT default to your training cutoff year.]\n"
        f"  - Date         : {now.strftime('%Y-%m-%d')} (Gregorian / 公曆)\n"
        f"  - Day of week  : {weekday_en} / {weekday_zh}\n"
        f"  - ISO 8601     : {iso}\n"
        f"  - Timezone     : {tz}\n"
        "  - If asked about Chinese lunar (農曆) date, derive it from the "
        "Gregorian date above — do NOT pull a year from training data. "
        "If you can't compute it reliably, say so honestly.\n"
        "\n"
    )


def _plan_prompt_for_mode(mode: str, model_name: str = "") -> str:
    """Pick the plan-system prompt that matches the session's mode and
    bend its strictness to the model's tier (Adaptive Prompts #1).

    Every prompt is prepended with the intent-first decomposition header
    (Issue #1) so over-decomposition of structured user inputs gets
    collapsed BEFORE the model starts listing subtasks. Applies to all
    tiers — Gemma 4 e2b through Claude all benefit from being asked to
    write the user's intent in one sentence first.
    """
    from model_profile import model_tier
    base = _PLAN_SYSTEM_CHAT if mode == "chat" else (
        _PLAN_SYSTEM_ANALYTIC if mode == "analytic" else _PLAN_SYSTEM_CODE
    )
    tier = model_tier(model_name)
    # Current-time header goes FIRST so every model (regardless of tier)
    # has a real-clock anchor before reading anything else.
    time_hdr = _current_time_header()
    if tier == "strict":
        return time_hdr + _PLAN_INTENT_FIRST_HEADER + _PLAN_STRICT_HEADER + base
    # Mid-range and large models keep the existing prompt — the JSON
    # contract is already restrictive enough; loosening it doesn't help.
    return time_hdr + _PLAN_INTENT_FIRST_HEADER + base


def _is_local_only() -> bool:
    """Thin wrapper so orchestrator code reads cleanly at call sites."""
    from config import is_session_local_only
    return is_session_local_only()


# ── Response-framing language detection ─────────────────────────────────
# Server-side label strings the orchestrator wraps around tool logs +
# confirmation blocks USED to be Chinese-only (the project started as
# a zh-TW codebase). User reports seeing those Chinese labels even when
# they typed in English — confusing. Fix: pick the framing language
# from the user's GOAL text, not a global setting. So:
#   English goal  → "[Subtask 0]", "Status:", "Confirmation status"
#   Chinese goal  → "[子任務 0]", "狀態:", "確認狀態"
#
# Heuristic: CJK chars vs total alpha+CJK chars. > 30% CJK → Chinese.
# This handles "build me a 報告 dashboard" reasonably (mostly English →
# English framing). Tiny goals (e.g. "ls") default to English.
def _framing_lang(goal: str) -> str:
    if not goal:
        return "en"
    cjk = 0
    total = 0
    for ch in goal:
        if "一" <= ch <= "鿿":
            cjk += 1
            total += 1
        elif ch.isalpha():
            total += 1
    if total < 3:
        return "en"
    return "zh" if cjk / total > 0.30 else "en"


# Labels rendered into the user-visible summary + confirmation block.
# Keep keys identical across locales so the call sites stay flat.
_FRAMING: Dict[str, Dict[str, str]] = {
    "en": {
        "subtask_prefix":    "Subtask",
        "status":            "Status",
        "result":            "Result",
        "error":             "Error",
        "no_output":         "no output",
        "task_completed":    "Task completed",
        "confirmation_header": "\n\n---\n📋 **Confirmation status**",
        "container_status":  "Container status (this task)",
        "docker_status":     "Docker container status",
        "workspace_contents": "Workspace contents",
        "installed_pkgs":    "Installed packages (top 20)",
        "node_packages":     "Node packages",
    },
    "zh": {
        "subtask_prefix":    "子任務",
        "status":            "狀態",
        "result":            "結果",
        "error":             "錯誤",
        "no_output":         "無輸出",
        "task_completed":    "任務已完成",
        "confirmation_header": "\n\n---\n📋 **確認狀態**",
        "container_status":  "本任務容器狀態",
        "docker_status":     "Docker 容器狀態",
        "workspace_contents": "工作目錄內容",
        "installed_pkgs":    "已安裝套件（前20個）",
        "node_packages":     "Node 套件",
    },
}


def _L(goal: str, key: str) -> str:
    """Look up a framing label in the user's goal language."""
    return _FRAMING[_framing_lang(goal)][key]


# ── Empty-artifact hallucination guard (#A) ─────────────────────────────
# Small models on long-context tool-use tasks routinely "complete"
# work by writing a plausible summary instead of actually producing
# the requested file. The deliverable judge (line ~786 area in
# _run_subtask) catches THIS-FILE-IS-WRONG cases, but is silent when
# NO file was produced at all. _looks_like_file_producing_task is the
# heuristic that decides whether the absence of artifacts should be
# treated as a failure.
#
# Conservative by design: explain/describe/tell-me tasks are skipped
# entirely (they're answer-shaped, not file-shaped), and we require
# BOTH a strong production verb AND a concrete artifact noun /
# extension. So "explain how to build a website" won't trigger.
_FILE_PRODUCTION_VERBS = (
    # English
    "build", "create", "implement", "write", "generate", "render",
    "export", "develop", "produce", "make", "save", "scaffold",
    # 中文
    "建立", "建構", "建造", "撰寫", "產生", "產出", "輸出",
    "開發", "編寫", "做出", "寫一", "寫個", "寫成",
)
_ARTIFACT_NOUNS = (
    # English (multi-char so they don't collide with verb roots)
    "file", "script", "code", "report", " app", "application",
    "website", "web app", "page", "dashboard", "module",
    "function", "class ", "test ", "config", "component",
    "deck", "slide", "presentation", "spreadsheet", "csv",
    # 中文
    "檔案", "腳本", "程式", "程序", "代碼", "代码",
    "報告", "报告", "報表", "报表",
    "網頁", "网页", "網站", "网站", "應用", "应用",
    "頁面", "页面", "儀表板", "仪表板",
)
_ARTIFACT_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".htm", ".css",
    ".csv", ".json", ".yaml", ".yml", ".md", ".sh", ".sql",
    ".java", ".go", ".rs", ".cpp", ".hpp", ".rb", ".php",
    ".swift", ".kt", ".scala", ".dart", ".toml", ".dockerfile",
)
# Multi-word command phrases that are unambiguously side-effect-producing
# regardless of what surrounds them in the goal. Catches user-reported
# "I say tell me what's this project and docker build ..." where the
# verb+noun pair didn't match (no concrete artifact noun) but the
# explicit shell phrase makes the production intent obvious.
# Lowercased; matched via substring-in-string for speed (these are
# all distinctive enough that false positives are very unlikely —
# nobody writes "tell me about the history of docker build" expecting
# pure prose).
_EXECUTION_PHRASES = (
    "docker build", "docker run", "docker compose up", "docker compose build",
    "docker-compose up", "docker-compose build",
    "npm run build", "npm install", "yarn install", "yarn build",
    "pnpm install", "pnpm build",
    "pip install", "uv install", "uv pip install", "uv run",
    "make build", "make install",
    "go build", "go install", "cargo build", "cargo install", "cargo run",
    "git clone", "git pull", "git push",
    "terraform apply", "kubectl apply",
)
# Answer-shaped task starts — exclude these entirely. Match on the
# first ~30 chars so we catch "describe how..." but not "write a
# function that describes...".
_ANSWER_TASK_STARTS = (
    "explain", "describe", "summarise", "summarize", "tell me",
    "what is", "what are", "how does", "how do",
    "解釋", "解释", "說明", "说明", "描述",
    "告訴我", "告诉我", "請問", "请问",
)


def _looks_like_file_producing_task(description: Optional[str]) -> bool:
    """True iff the subtask description strongly implies a file should
    appear on disk. Used to detect hallucinated successes — see the
    callsite in _run_subtask for the full rationale.

    Issue #2A fix (2026-05-17): strong-signal-precedes ordering.
    Previously this checked the answer-shaped-start blacklist FIRST,
    which mis-classified goals like "I say tell me what's this project
    and docker build and tell me the service URL(s)" as pure-answer
    tasks → skipped the guard entirely. User-reported case where
    Mixed mode failed to escalate because the empty-artifact check
    never ran.

    New order:
      1. Compute production-verb + artifact-noun/ext signals first
      2. If BOTH are present → return True (strong signal wins, even
         if the goal starts with "tell me / explain / describe")
      3. Otherwise apply the answer-shape blacklist as before — pure
         "explain X" / "what is Y" tasks still skip the guard
    """
    if not description:
        return False
    text = description.strip()
    lower = text.lower()

    has_verb = any(v in lower for v in _FILE_PRODUCTION_VERBS)
    has_noun = any(n in lower for n in _ARTIFACT_NOUNS)
    has_ext = any(ext in lower for ext in _ARTIFACT_EXTENSIONS)
    has_exec_phrase = any(p in lower for p in _EXECUTION_PHRASES)

    # Strong signal #1: production verb + concrete artifact / file ext
    # → treat as file-producing regardless of how the goal opens.
    if has_verb and (has_noun or has_ext):
        return True

    # Strong signal #2: explicit multi-word execution phrase
    # ("docker build", "npm install", "git clone", …) — these have
    # unmistakable side effects on disk / container state, so any goal
    # mentioning them is a production task even when it ALSO contains
    # "tell me / explain" wrappers (compound goals are common).
    if has_exec_phrase:
        return True

    # Otherwise the answer-shape blacklist applies — "explain X" /
    # "describe Y" without any production signal genuinely IS an
    # answer task.
    head = lower[:40]
    if any(head.startswith(start) for start in _ANSWER_TASK_STARTS):
        return False

    return False  # weak signal and no answer-shape match either


class Orchestrator:
    """Gemma-powered orchestrator that plans, dispatches, and aggregates."""

    def __init__(
        self,
        tracker: Tracker,
        executor: Executor,
        escalation: EscalationAgent,
        skill_factory: SkillFactory,
        memory: Optional["MemoryManager"] = None,
    ) -> None:
        self.tracker      = tracker
        self.executor     = executor
        self.escalation   = escalation
        self.skill_factory = skill_factory
        self.memory       = memory
        self._http        = httpx.AsyncClient(timeout=180)
        self._consecutive_failures: Dict[str, int] = {}

    async def close(self) -> None:
        await self._http.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, req: TaskRequest) -> TaskResult:
        """Full lifecycle: plan → execute → escalate → summarise."""
        logger.info("Orchestrator starting task %s: %s", req.id, req.goal)

        # Apply per-session workspace override + privacy flag. Both are
        # ContextVars so concurrent tasks in other sessions keep their
        # own setting. Values are NOT manually cleared — asyncio
        # ContextVars are task-local, so they naturally scope to this run.
        from config import set_session_workspace, set_session_local_only
        session_ws: Optional[str] = None
        session_local_only = False
        session_mode = "code"
        if req.session_id:
            try:
                sess = await self.tracker.get_session(req.session_id)
                if sess:
                    session_ws = sess.get("workspace_dir") or None
                    session_local_only = bool(sess.get("local_only"))
                    session_mode = (sess.get("mode") or "code").lower()
            except Exception:  # noqa: BLE001
                pass
        set_session_workspace(session_ws)
        set_session_local_only(session_local_only)

        # Bind session_id to the per-task ContextVar that tool
        # implementations read. Set at the top of run() so the whole
        # task lifecycle — planning, executor turns, escalation,
        # summary synthesis — sees the same session. ContextVar is
        # asyncio-task-local; concurrent orchestrator.run() calls in
        # other sessions don't see each other's binding.
        # Tools that need to look up per-session state (db_tool's
        # engine builder reads the session's attached DB URL via
        # tracker) just call tools._context.get_session_id() —
        # no extra plumbing through executor/registry needed.
        from tools._context import set_session_id as _ot_set_session
        _ot_set_session(req.session_id or "")
        if session_ws:
            logger.info(
                "Task %s using session-specific workspace: %s",
                req.id, session_ws,
            )
        if session_local_only:
            logger.info(
                "Task %s is local-only — Claude escalation will be skipped "
                "regardless of confidence.",
                req.id,
            )

        # Persist the task
        await self.tracker.create_task(req)
        await self.tracker.update_task_status(req.id, TaskStatus.RUNNING)

        # ── 0a. Chat-mode short-circuit (highest-priority fast path) ──
        # When the user has explicitly picked Chat mode for the session,
        # we already know this isn't going to involve tools — skip the
        # intent-classifier call entirely (saves a whole LLM round-trip,
        # ~500 ms-2 s on a local Gemma) and go straight to streaming
        # the answer.
        #
        # Why this is safe: session_mode == "chat" is an explicit user
        # signal, far stronger than the classifier's confidence number.
        # Anyone who *did* want code execution would have picked Code
        # mode in the session row; the LLM has no agency to override
        # the user's choice.
        #
        # User-facing impact: a question like "Python lambda 怎麼用"
        # in a chat session goes from 2 LLM calls (classifier + answer)
        # to 1 (answer only) — typically the difference between feeling
        # snappy and feeling sluggish on a local model.
        if session_mode == "chat":
            logger.info(
                "Chat-mode fast-path engaged (skipped intent classifier): task=%s",
                req.id,
            )
            try:
                return await self._fast_chat_response(
                    req, session_mode,
                    # Synthetic intent dict — fast-path persists the
                    # classifier's confidence on the synthetic subtask
                    # so Usage-tab stats can distinguish path types.
                    # 0.95 reflects the unambiguous nature of the
                    # signal (user explicitly chose chat).
                    {
                        "needs_tools":  False,
                        "is_pure_chat": True,
                        "confidence":   0.95,
                        "rationale":    "explicit chat session mode",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                # If the streamlined path blows up for any reason (the
                # _orchestrator_complete LLM call failed, network blip,
                # model unloaded mid-call), fall through to the full
                # plan → execute flow rather than failing the task.
                logger.warning(
                    "Chat-mode fast-path failed (falling back to full flow): %s",
                    exc,
                )

        # ── 0b. Intent classifier (fast-path for pure-chat goals in
        # code / analytic sessions) ─────────────────────────────────────
        # Before paying for plan → execute → summary on goals that
        # legitimately don't need tools (questions, explanations,
        # opinions), classify the goal in ~100 ms and short-circuit
        # straight to a single-LLM-turn answer. Cuts ~25 s off the
        # latency for "what is X" style questions that were previously
        # running a full empty tool loop. Skipped above for explicit
        # chat-mode sessions (where we know without asking).
        #
        # Conservative: only fires when the classifier is confident
        # (>= 0.7) AND says BOTH needs_tools=False AND is_pure_chat=True.
        # Any uncertainty falls through to the normal plan→execute flow.
        if getattr(config, "intent_classifier_enabled", True):
            try:
                intent = await self._classify_intent(req.goal, session_mode)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intent classifier failed (non-fatal): %s", exc)
                intent = None
            if (intent
                    and not intent.get("needs_tools", True)
                    and intent.get("is_pure_chat", False)
                    and float(intent.get("confidence", 0.0)) >= 0.7):
                logger.info(
                    "Fast-path engaged: intent=chat conf=%.2f reason=%r",
                    intent["confidence"],
                    intent.get("rationale", "")[:80],
                )
                # Fast-path can bail (empty model response) by raising —
                # that's a signal to fall through to the full plan→execute
                # flow, NOT to fail the whole task. Without this try the
                # RuntimeError propagated and the dispatch died with
                # "fast-path empty response".
                try:
                    return await self._fast_chat_response(
                        req, session_mode, intent,
                    )
                except RuntimeError as exc:
                    logger.info(
                        "Fast-path bailed (%s) — continuing to full flow", exc
                    )
                    # fall through to the planner below

        try:
            # 1. Plan (with optional memory context)
            subtasks = await self._plan(req)
            # Defensive pass: merge bare `cd X` subtasks into the next
            # subtask with `&&` so the cd actually takes effect.
            subtasks = self._flatten_cd_subtasks(subtasks)
            for st in subtasks:
                await self.tracker.create_subtask(st)

            # 2. Execute subtasks sequentially (could be made parallel later)
            skills_used: list[str] = []
            new_skills:  list[str] = []

            mode_value = req.mode.value if hasattr(req.mode, "value") else str(req.mode)
            total_subtasks = len(subtasks)
            for st in subtasks:
                st = await self._run_subtask(st, req.context, mode=mode_value)

                # ── Live progress pill (Sprint 2B) ────────────────────
                # Broadcast cumulative state so the chat UI's pill can
                # show "subtask 3/5 · 🎯 0.85 · 💰 $0.043". Cheap —
                # one tracker aggregate query per subtask, fire-and-
                # forget WS broadcast.
                try:
                    usage = await self.tracker.get_task_usage(req.id)
                    await self.executor._push_event({
                        "type":       "subtask.progress",
                        "task_id":    req.id,
                        "order":      (st.order or 0) + 1,
                        "total":      total_subtasks,
                        "confidence": float(getattr(st, "confidence", 0.0) or 0.0),
                        "status":     getattr(st.status, "value", str(st.status)),
                        "tokens_in":  usage["tokens_in"],
                        "tokens_out": usage["tokens_out"],
                        "cost_usd":   usage["cost_usd"],
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.debug("subtask.progress broadcast failed: %s", exc)

                if st.skill_hint:
                    skills_used.append(st.skill_hint)

                # Track consecutive failures per task
                if st.status == TaskStatus.FAILED:
                    self._consecutive_failures[req.id] = (
                        self._consecutive_failures.get(req.id, 0) + 1
                    )
                else:
                    self._consecutive_failures[req.id] = 0

                # Hard abort if too many sequential failures
                if (
                    self._consecutive_failures.get(req.id, 0)
                    >= config.escalation_failure_limit
                ):
                    logger.warning(
                        "Task %s hit failure limit — aborting remaining subtasks.",
                        req.id,
                    )
                    break

            # 3. Collect any newly created skills
            all_skills = await self.skill_factory.list_all_skills()
            new_skills  = [s.name for s in all_skills if s.success_count == 0]

            # 4. Build structured tool log from all subtask records.
            # Framing labels switch on the user's goal language so an
            # English-typing user doesn't see Chinese "子任務 0 / 狀態:"
            # leaking into the final summary (those labels go into the
            # LLM-summary prompt and the model mirrors them verbatim).
            _g = req.goal or ""
            tool_log_lines: List[str] = []
            for st in subtasks:
                tool_log_lines.append(
                    f"[{_L(_g, 'subtask_prefix')} {st.order}] {st.description}"
                )
                tool_log_lines.append(f"  {_L(_g, 'status')}: {st.status.value}")
                tool_log_lines.append(
                    f"  {_L(_g, 'result')}: "
                    f"{(st.result or _L(_g, 'no_output'))[:500]}"
                )
                if st.error:
                    tool_log_lines.append(f"  {_L(_g, 'error')}: {st.error}")
            tool_log_text = "\n".join(tool_log_lines)

            had_escalation = any(
                getattr(st, 'status', None) == TaskStatus.ESCALATED for st in subtasks
            )
            results_texts = [st.result or "" for st in subtasks if st.result]

            # 5. Generate summary: Claude if escalated, Gemma otherwise.
            #
            # Chat mode is a special case — there's no "task" to summarise,
            # the user said "你好" and the executor said "你好！👋". Running
            # the summariser produces an awkward "任務已完成：你好" wrapper
            # (or worse, the raw tool_log fallback when Gemma returns
            # empty). For chat we just hand back the executor's raw text.
            if session_mode == "chat" and not had_escalation and results_texts:
                # Concatenate any subtask results — usually 1 in chat mode,
                # but defensive in case the planner emitted more.
                summary = "\n\n".join(t.strip() for t in results_texts if t.strip())
            elif had_escalation or not results_texts:
                # 只有真的升級過才用 Claude 做 summary
                summary = await self.escalation.synthesize_summary(
                    req.goal, results_texts, task_id=req.id
                )
            else:
                # 全部地端完成 → 用 Gemma 讀取工具記錄，產出中文完成報告
                summary = await self._gemma_summarize(req.goal, tool_log_text, task_id=req.id)

            # 6. 任務完成後自動執行確認指令，把結果附加到 summary
            #    Only runs for Code-mode deploys — Chat/Analytic don't
            #    need a shell-based "did docker come up?" block, and
            #    surfacing a raw `ls -la ~/` in the middle of an analysis
            #    report is noise the user explicitly complained about.
            confirmation_output = await self._run_confirmation_checks(
                req.goal, subtasks=subtasks, session_mode=session_mode,
            )
            if confirmation_output:
                summary += confirmation_output

            overall_status = self._derive_status(subtasks)
            await self.tracker.update_task_status(req.id, overall_status, summary)

            # 5. Persist this task's outcome to long-term memory
            if self.memory is not None:
                try:
                    await self.memory.summarize_and_store(
                        task_id=req.id,
                        goal=req.goal,
                        final_output=summary,
                        session_id=req.session_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Memory store failed (non-fatal): %s", exc)

            # 6. Auto-detect recurring goal patterns → promote to skill
            #    (only fires on a successfully completed task to avoid
            #    learning broken patterns). Async fire-and-forget — the
            #    user shouldn't have to wait for skill synthesis to see
            #    their task result.
            if (overall_status == TaskStatus.COMPLETED
                    and self.memory is not None
                    and self.memory.is_available
                    and not _is_local_only()):
                # Schedule as background task — embedding query + LLM
                # synthesis takes 1-3 s and shouldn't delay the return.
                asyncio.create_task(
                    self._maybe_promote_pattern_to_skill(req)
                )

            return TaskResult(
                task_id=req.id,
                status=overall_status,
                summary=summary,
                subtasks=subtasks,
                skills_used=list(set(skills_used)),
                new_skills_created=new_skills,
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Orchestrator error for task %s: %s", req.id, exc)
            await self.tracker.update_task_status(
                req.id, TaskStatus.FAILED, str(exc)
            )
            raise

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _plan(self, req: TaskRequest) -> List[SubTask]:
        """Ask Gemma to decompose the goal into SubTasks.
        Injects relevant memory context into the system prompt when available."""
        active_skills = await self.skill_factory.list_active_skills()
        skill_names   = [s.name for s in active_skills]

        # ── Retrieve long-term memory context ────────────────────────────────
        # Scoped to the request's session_id so cross-project memories don't
        # leak in (e.g. prior mold-harvester tasks bleeding into a fresh
        # Pixelle-Video deployment).
        memory_ctx = ""
        if self.memory is not None:
            try:
                memory_ctx = await self.memory.get_context_for_task(
                    req.goal, session_id=req.session_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Memory retrieval failed (non-fatal): %s", exc)

        # Pick the plan prompt based on the session's mode. This is the key
        # insight behind the mode selector: instead of one prompt trying to
        # classify the task, we hand Gemma a prompt already specialised for
        # the user's declared intent.
        mode_value = req.mode.value if hasattr(req.mode, "value") else str(req.mode)
        base_prompt = _plan_prompt_for_mode(mode_value, config.gemma_model)

        # For Code / Analytic modes, tell Gemma what the agent workspace is
        # so the shell plan uses a known directory (git clones land there
        # by default, not scattered across /tmp or the server's cwd). Chat
        # mode doesn't touch the filesystem so we skip this.
        workspace_hint = ""
        if mode_value in ("code", "analytic"):
            try:
                # Use effective workspace so Gemma sees THIS session's path
                # (the user may have pointed this session at a specific
                # project folder; telling Gemma "clone into agent-workspace"
                # would be wrong in that case).
                from config import effective_workspace_dir
                ws = os.path.abspath(effective_workspace_dir())
                ws_basename = os.path.basename(ws.rstrip("/"))
                workspace_hint = (
                    f"\n\n【工作目錄 — 非常重要，常踩坑，一定要讀完】\n"
                    f"  Workspace 絕對路徑：{ws}\n"
                    f"  所有 shell 指令的 cwd 預設就是這個路徑（不用 cd 進去）。\n"
                    f"  此路徑以外的一切**不存在於你的世界**，別去 reach 出去。\n"
                    f"\n"
                    f"  【規則 1 — 有 git URL 就一定要 clone，不准跳過】\n"
                    f"  若任務裡有 github.com / gitlab.com 之類的 URL，**第一個子任務**\n"
                    f"  **必須**是 `git clone <URL>`，不管記憶裡說什麼。記憶可能壞、\n"
                    f"  可能舊、可能路徑根本不對。**先 clone 再說**。\n"
                    f"    ✅ [{{description: 'git clone https://github.com/x/y'}}, {{...}}]\n"
                    f"    ❌ 第一步直接 docker_project_detect（repo 可能不存在）\n"
                    f"\n"
                    f"  【規則 2 — 禁止 `..` 路徑】\n"
                    f"  絕對不要用 `../X` / `./../X` 這種會跳出 workspace 的路徑。\n"
                    f"  所有 clone 下來的專案都在 workspace 底下，直接用 `<repo>/` 就好。\n"
                    f"    ❌ `docker_project_detect(working_dir='../Pixelle-Video')`\n"
                    f"    ❌ `cd ../Pixelle-Video && ...`\n"
                    f"    ✅ `docker_project_detect(working_dir='Pixelle-Video')`\n"
                    f"    ✅ `cd Pixelle-Video && ...`\n"
                    f"\n"
                    f"  【規則 3 — 禁止 workspace 前綴】\n"
                    f"  也不要在 shell 指令寫 `{ws_basename}/` 前綴 —— shell 已經\n"
                    f"  在 `{ws_basename}` 裡面，再 cd 會找 `{ws_basename}/{ws_basename}/`\n"
                    f"  這個不存在的雙層路徑，cd 失敗，`&&` 後面全部靜默跳過。\n"
                    f"    ❌ `cd ./{ws_basename}/y && ...`\n"
                    f"    ❌ `cd {ws_basename}/y && ...`\n"
                    f"    ✅ `cd y && ...`\n"
                    f"\n"
                    f"  【規則 4 — 驗證要真的看到服務】\n"
                    f"  `docker compose up` 之後必須 `docker compose ps` 確認。\n"
                    f"  若只有表頭 `NAME STATUS SERVICE` 沒有服務 row，**那是失敗**\n"
                    f"  （通常 cwd 錯了或 compose 檔根本沒載入），不是成功。\n"
                    f"  下一步用 docker_diagnose 或 compose_validate 找根因。\n"
                    f"\n"
                    f"  【規則 5 — 看到 `no such file` / `cannot list` 立刻查上游】\n"
                    f"  這代表某個前置步驟沒做或做錯了。不要無視繼續往下跑。\n"
                    f"\n"
                    f"  【規則 6 — 絕對禁止對 OpenTeddy 本體動手術】\n"
                    f"  OpenTeddy 自己的原始碼目錄**不是**你的工作目錄。\n"
                    f"  不要把 working_dir 設成 OpenTeddy 的專案根目錄，\n"
                    f"  也不要 cd 進去跑 docker compose / rm / mv 等會改\n"
                    f"  變狀態的指令 —— 會搞壞正在跑的 agent。只有 `{ws_basename}/`\n"
                    f"  底下的子目錄才是合法沙盒。shell_tool 會硬性擋下違反\n"
                    f"  這條規則的指令，計劃階段就要避開。\n"
                    f"\n"
                    f"  【規則 7 — 相信 session workspace，不要自作主張傳絕對路徑】\n"
                    f"  你看到的 workspace 路徑（{ws}）是這個 session 專屬的。\n"
                    f"  下面那些 tool / shell 指令，working_dir 一律用**相對路徑**\n"
                    f"  （就寫子目錄名，例如 `worldmonitor`、或留空 = 直接在 workspace）。\n"
                    f"  **禁止**在 tool call 裡把 working_dir 填絕對路徑跳出這個範圍\n"
                    f"  （例如 /home/user/OpenTeddy、/home/user/otherproject）—— 那是\n"
                    f"  繞過 session 設定，使用者會很困擾。\n"
                )
            except Exception:  # noqa: BLE001
                pass

        system_prompt = base_prompt + workspace_hint
        if memory_ctx:
            system_prompt = memory_ctx + "\n\n" + system_prompt

        prompt = (
            f"Goal: {req.goal}\n\n"
            f"Available skills: {json.dumps(skill_names)}\n\n"
            "Output the sub-task plan now."
        )
        # Routes to cloud LLM when llm_mode='cloud' so the user with
        # no Ollama installed can still get a plan. Otherwise stays on
        # Gemma. See _orchestrator_complete for fallback behaviour.
        raw_plan = await self._orchestrator_complete(
            prompt, system_prompt,
            task_id=req.id,
            task_description=f"[plan] {req.goal[:100]}",
        )
        subtasks  = self._parse_plan(raw_plan, req.id, req.goal)

        if not subtasks:
            # Fallback: single subtask = the whole goal
            subtasks = [
                SubTask(
                    parent_task_id=req.id,
                    description=req.goal,
                    agent=AgentRole.EXECUTOR,
                    order=0,
                )
            ]
        logger.info("Planned %d subtasks for task %s", len(subtasks), req.id)
        return subtasks

    @staticmethod
    def _flatten_cd_subtasks(subtasks: List[SubTask]) -> List[SubTask]:
        """Merge ``cd <path>`` subtasks into the following subtask.

        Each shell command runs in its own subprocess, so a bare ``cd``
        subtask has no lasting effect on subsequent subtasks. When the
        planner (Gemma) splits a directory switch and its operation into
        two separate subtasks — a classic mistake — collapse them into a
        single ``cd <path> && <next-command>`` subtask.

        A trailing bare cd with nothing after it is dropped (no-op).
        """
        if not subtasks:
            return subtasks

        flattened: List[SubTask] = []
        i = 0
        n = len(subtasks)
        while i < n:
            st = subtasks[i]
            match = _BARE_CD_RE.match(st.description or "")
            if match and i + 1 < n:
                path = match.group("path")
                nxt = subtasks[i + 1]
                next_desc = _LEADING_VERB_RE.sub("", nxt.description or "").strip()
                nxt.description = f"執行 cd {path} && {next_desc}"
                logger.info(
                    "Flattening bare 'cd %s' subtask into next: %r",
                    path, nxt.description[:120],
                )
                i += 1  # skip the bare cd; next iteration handles the merged next
                continue
            if match and i + 1 == n:
                logger.info(
                    "Dropping trailing bare 'cd %s' subtask (no-op).",
                    match.group("path"),
                )
                i += 1
                continue
            flattened.append(st)
            i += 1

        for idx, st in enumerate(flattened):
            st.order = idx
        return flattened

    async def _run_subtask(
        self, st: SubTask, context: Dict[str, Any], mode: str = "code",
    ) -> SubTask:
        """Execute one subtask with retry; only escalate to Claude after all retries fail.

        Each attempt is wrapped with ``asyncio.wait_for(timeout=subtask_timeout)``
        so that a hung local model (Qwen tool-call freeze) is detected and
        immediately escalated to Claude instead of blocking forever.

        On success, appends a verification result if a matching check exists.
        """
        # ── Cloud mode short-circuit ─────────────────────────────────────────
        # When Settings → LLM Mode is set to 'cloud', skip Qwen entirely
        # and hand the subtask straight to escalation.resolve() (= cloud
        # LLM via the configured provider). Saves the 5-10s Qwen first-
        # attempt latency on every subtask and prevents the "Qwen wrote
        # a confident-looking wrong answer" failure mode for users who
        # explicitly want max quality.
        #
        # Session-level local_only still wins via is_cloud_mode() — a
        # privacy-mode session inside a cloud-mode app stays local.
        # Planning (Gemma) currently still runs locally; full-cloud
        # planning is on the roadmap. The user gets a single round-trip
        # to the cloud per subtask which is the dominant cost saver.
        try:
            from config import is_cloud_mode as _is_cloud_mode
            cloud_routing = _is_cloud_mode()
        except Exception:  # noqa: BLE001
            cloud_routing = False

        if cloud_routing:
            logger.info(
                "Subtask %s — cloud mode active, bypassing Qwen and routing "
                "directly to escalation.resolve()",
                st.id,
            )
            try:
                return await self.escalation.resolve(st, context)
            except Exception as exc:  # noqa: BLE001
                # Cloud failed → fall through to local executor as a
                # safety net rather than failing the whole subtask. The
                # user picked cloud-first, not cloud-only; falling back
                # to local is a better UX than a hard error.
                logger.warning(
                    "Cloud-mode escalation failed for %s (%s) — falling "
                    "back to local executor", st.id, exc,
                )

        max_local_retries = getattr(config, 'escalation_failure_limit', 3)
        subtask_timeout   = getattr(config, 'subtask_timeout', 120)
        escalation_threshold = getattr(config, 'escalation_confidence_threshold', 0.6)

        # Subtasks that contain a long-running docker build get a much
        # bigger ceiling (1 hour) so the orchestrator doesn't cancel them
        # while shell_tool's silence-timeout is still happily watching
        # the build produce progress. Without this, `docker compose up
        # -d --build` on a fresh repo routinely tripped the 900s
        # subtask wrapper even though the build itself was healthy.
        desc_lower = (st.description or "").lower()
        effective_sub_timeout = subtask_timeout
        if any(kw in desc_lower for kw in
               ("docker compose up", "docker compose build", "docker-compose up",
                "docker build")):
            effective_sub_timeout = max(subtask_timeout, 3600)  # at least 1 hour
            logger.info(
                "Subtask %s looks docker-heavy — bumping wall timeout %ds → %ds "
                "(silence-timeout still watches for real hangs)",
                st.id, subtask_timeout, effective_sub_timeout,
            )

        for attempt in range(max_local_retries):
            try:
                st = await asyncio.wait_for(
                    self.executor.execute(st, context, mode=mode),
                    timeout=float(effective_sub_timeout),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Subtask %s timed out after %ds on attempt %d/%d — escalating to Claude.",
                    st.id, effective_sub_timeout, attempt + 1, max_local_retries,
                )
                st.status = TaskStatus.FAILED
                st.error  = (
                    f"地端模型執行超時（>{effective_sub_timeout}s）"
                    + (" — 本地端模式，不升級 Claude" if _is_local_only() else "，自動升級到 Claude")
                )
                await self.tracker.update_subtask(st)
                if _is_local_only():
                    return st  # respect privacy guardrail
                st = await self.escalation.resolve(st, context)
                return st

            confidence = getattr(st, 'confidence', 1.0)

            if st.status != TaskStatus.FAILED and confidence >= escalation_threshold:
                # 地端成功 → 嘗試執行驗證指令
                verification = await self._verify_subtask(st)
                if verification:
                    text, is_healthy = verification
                    st.result = (st.result or "") + f"\n\n[驗證結果]\n{text}"
                    if not is_healthy:
                        logger.info(
                            "Subtask %s verification detected unhealthy state "
                            "— escalating to Claude.",
                            st.id,
                        )
                        st.status = TaskStatus.FAILED
                        st.error = (
                            "驗證發現容器不健康 / 未正常啟動"
                            + ("（本地端模式，不升級 Claude）" if _is_local_only()
                               else "，自動升級 Claude 診斷")
                        )
                        await self.tracker.update_subtask(st)
                        if _is_local_only():
                            return st
                        st = await self.escalation.resolve(st, context)
                        return st

                # ── Per-step deliverable verification (#2) ───────────
                # Pull the artifacts the executor produced for this
                # subtask, ask Qwen if any of them actually fulfils the
                # goal (vs. being a description / report / plan ABOUT
                # the goal — the snake.html "fake completion" pattern).
                # Last shot at retry — bumps the failure counter and
                # falls through to the loop's retry path.
                artifacts = self.executor.pop_subtask_artifacts(st.id)

                # ── #A: empty-artifact hallucination guard ───────────
                # Catches the "Qwen claimed success without writing
                # anything" failure mode the deliverable judge below
                # CAN'T catch (the judge needs a file to look at).
                # User-reported on macOS desktop: "build a GitHub
                # trending web app" returned a confident success
                # summary with workspace size = 0 bytes.
                #
                # Heuristic: code/analytic mode + description has a
                # strong production verb + concrete artifact noun (or
                # a known file extension) + zero new artifacts.
                # Conservative — answer-type subtasks ("explain X",
                # "describe Y") are explicitly skipped so they don't
                # get false-failed.
                if (mode in ("code", "analytic")
                        and not artifacts
                        and _looks_like_file_producing_task(st.description)):
                    logger.info(
                        "Subtask %s should have produced a file but "
                        "workspace got 0 new artifacts — clamping "
                        "confidence and retrying.",
                        st.id,
                    )
                    st.status = TaskStatus.FAILED
                    st.error = (
                        "Task description suggested building / creating "
                        "a concrete file, but no file was produced "
                        "(workspace got 0 new artifacts). The model "
                        "may have answered with prose instead of "
                        "writing the artifact. Retrying with stronger "
                        "guidance, then escalating if it fails again."
                    )
                    st.confidence = 0.3
                    await self.tracker.update_subtask(st)
                    continue  # → next retry attempt, eventually escalates

                deliverable = await self._verify_deliverable(st, artifacts)
                if deliverable:
                    judge_text, judge_ok = deliverable
                    st.result = (st.result or "") + f"\n\n{judge_text}"
                    if not judge_ok:
                        logger.info(
                            "Subtask %s deliverable judge FAILED — "
                            "retrying with feedback.", st.id,
                        )
                        st.status = TaskStatus.FAILED
                        st.error = (
                            "Deliverable did not match the goal — looks like "
                            "a description / placeholder rather than the "
                            "requested artifact. " + judge_text
                        )
                        st.confidence = 0.3   # force the loop to retry
                        await self.tracker.update_subtask(st)
                        # Fall through to retry — DON'T return success.
                        continue

                await self.tracker.update_subtask(st)
                return st

            logger.info(
                "Subtask %s local attempt %d/%d failed (conf=%.2f), retrying...",
                st.id, attempt + 1, max_local_retries, confidence,
            )

        # 所有重試都失敗 → 才升級到 Claude（本地端模式除外）
        if _is_local_only():
            logger.info(
                "Subtask %s local-only — skipping Claude escalation after "
                "%d failed attempts.",
                st.id, max_local_retries,
            )
            st.status = TaskStatus.FAILED
            st.error = (
                (st.error or "")
                + "\n\n[本地端模式] 地端重試用盡，依 session 設定不升級 Claude。"
            ).strip()
            await self.tracker.update_subtask(st)
            return st
        logger.info(
            "Subtask %s escalating to Claude after %d failed attempts.",
            st.id, max_local_retries,
        )
        st = await self.escalation.resolve(st, context)
        return st

    # ── Per-step deliverable verification (#2) ──────────────────────────
    # LLM-as-judge check: for tasks that produced a file, ask Qwen
    # whether the file content actually fulfils the original goal, vs.
    # being a description / report / placeholder ABOUT the goal.
    # This is the direct fix for the "small model wrote a Snake Game
    # report instead of a runnable game" failure mode the user saw.
    _DELIVERABLE_JUDGE_PROMPT = (
        "You are a strict QA reviewer for an AI agent. Decide if the "
        "produced FILE actually fulfils the user's GOAL — i.e. it is a "
        "real working artifact, NOT a report / outline / tutorial / "
        "description ABOUT the goal.\n\n"
        "Examples of FAIL:\n"
        "  • Goal asks for a working HTML game, file is a markdown / "
        "HTML report describing what such a game would contain "
        "(skeleton, placeholder, no real game loop / no addEventListener).\n"
        "  • Goal asks for a Python script, file is pseudocode in a "
        "code-fence with no runnable definitions.\n"
        "  • Goal asks for an analysis with numeric results, file just "
        "lists what the analysis WOULD compute.\n\n"
        "Reply STRICTLY with a JSON object — no markdown fences, no "
        "preamble — in this exact shape:\n"
        '  {"verdict": "PASS" | "FAIL", "reason": "<<= 30 words>"}\n'
        "Nothing else."
    )

    async def _verify_deliverable(
        self, st: SubTask, artifacts: List[Dict[str, Any]],
    ) -> Optional[Tuple[str, bool]]:
        """Ask Qwen if the largest produced deliverable artifact actually
        matches the subtask goal. Returns ``(reason, ok)`` or ``None`` if
        there was nothing to verify (no artifacts, none deliverable-shaped,
        or verification globally disabled)."""
        if not artifacts:
            return None
        # Master kill-switch — heavyweight on big models. Defaults true so
        # legacy users keep the safety net; flip in Settings → Performance
        # or via env VERIFICATION_ENABLED=false. The DGX-Spark/35B-class
        # user case is the canonical reason to disable.
        if not bool(getattr(config, "verification_enabled", True)):
            logger.debug("Subtask %s: verification disabled by config — skipping.", st.id)
            return None

        # DELIVERABLE-shaped extensions only. Raw data (.csv / .tsv /
        # .parquet / .json), config (.yaml / .toml / .ini / .xml), and
        # plain text logs are NOT included — they're typically inputs or
        # intermediates, not the user's actual deliverable. Including
        # them caused a regression where CSV-report plans got stuck:
        # the verifier picked the largest .csv artifact, judged it as
        # "not the report" (correctly!), and forced infinite retries.
        DELIVERABLE_EXTS = {
            ".html", ".htm",                # web reports / dashboards
            ".md",                          # markdown reports
            ".py", ".js", ".jsx", ".ts", ".tsx",   # implementation code
            ".css",                         # styling for HTML reports
            ".sh", ".sql",                  # scripted deliverables
        }
        candidates = [
            a for a in artifacts
            if any(a.get("path", "").lower().endswith(ext) for ext in DELIVERABLE_EXTS)
        ]
        if not candidates:
            logger.debug(
                "Subtask %s: no deliverable-shaped artifacts (had %d total) — "
                "skipping verification.", st.id, len(artifacts),
            )
            return None
        candidates.sort(key=lambda a: a.get("size_bytes") or 0, reverse=True)
        target = candidates[0]
        path = target.get("path") or ""

        # Cap content read at ~6 KB — enough to spot "this is a report
        # not a game", not enough to drown the verification call's
        # context. We sample HEAD only (most files give away their
        # nature in the first ~100 lines).
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(6_000)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Verifier could not read %s: %s", path, exc)
            return None

        user_msg = (
            f"GOAL: {st.description}\n\n"
            f"PRODUCED FILE: {os.path.basename(path)}\n"
            f"FILE CONTENT (first {len(content)} chars):\n"
            f"------\n{content}\n------"
        )
        try:
            resp = await self._http.post(
                f"{config.qwen_base_url}/api/chat",
                json={
                    "model":    config.qwen_model,
                    "messages": [{"role": "user", "content": user_msg}],
                    "system":   self._DELIVERABLE_JUDGE_PROMPT,
                    "stream":   False,
                    # Ollama's structured-output mode — forces the
                    # response.message.content to be valid JSON. Without
                    # this small thinking models love to wrap the
                    # verdict in essay-style commentary that's too
                    # ambiguous to parse reliably.
                    "format":   "json",
                    "options":  {
                        "temperature": 0.1,
                        "num_predict": 200,
                        "num_ctx":     int(getattr(config, "qwen_num_ctx", 16384)),
                    },
                    "keep_alive": getattr(config, "ollama_keep_alive", "24h"),
                },
                # 30s cap. The judge call only needs ~50 tokens of output
                # ("PASS" / "FAIL" + 30-word reason). On a healthy big
                # model that's <10s; if it blows past 30s something is
                # wrong (model reload, cold start, unrelated job hogging
                # GPU) and we're better off skipping verification than
                # blocking the whole plan for a minute per step.
                timeout=30,
            )
            resp.raise_for_status()
            msg = resp.json().get("message") or {}
            verdict_raw = (msg.get("content") or msg.get("thinking") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Deliverable verifier crashed: %s — skipping.", exc)
            return None

        # Parse the JSON response. Be defensive — Ollama's `format: json`
        # is reliable but not bulletproof, and some thinking models still
        # leak prose around the JSON object.
        ok: Optional[bool] = None
        reason = verdict_raw[:200]
        try:
            # Try to extract the first {...} block in case the model
            # decided to add commentary outside it anyway.
            match = re.search(r"\{[\s\S]*\}", verdict_raw)
            data = json.loads(match.group(0)) if match else json.loads(verdict_raw)
            v = str(data.get("verdict", "")).strip().upper()
            if v == "PASS":
                ok, reason = True,  data.get("reason") or "OK"
            elif v == "FAIL":
                ok, reason = False, data.get("reason") or "Did not match goal"
        except Exception:  # noqa: BLE001
            # JSON parse failed — fall back to keyword heuristics on
            # raw text. Better lenient than false-positive on a flaky
            # judge that just happens to disagree with our format.
            lower = verdict_raw.lower()
            fail_keywords = ("non-functional", "skeleton", "placeholder",
                              "describes", "report", "outline", "no actual",
                              "not implemented", "pseudocode", "no real")
            pass_keywords = ("complete", "functional", "fully implemented",
                              "working")
            if any(k in lower for k in fail_keywords):
                ok, reason = False, "verdict text suggests not a real artifact"
            elif any(k in lower for k in pass_keywords):
                ok, reason = True, "verdict text suggests artifact is real"

        if ok is None:
            # Genuinely couldn't tell — be lenient, treat as PASS so we
            # don't false-positive trip on a flaky judge.
            return None
        logger.info(
            "Deliverable verifier: %s (file=%s, reason=%s)",
            "PASS" if ok else "FAIL", os.path.basename(path), reason[:120],
        )
        return (f"[deliverable-judge] {os.path.basename(path)}: {reason}", ok)

    async def _verify_subtask(self, st: SubTask) -> Optional[Tuple[str, bool]]:
        """根據子任務類型執行驗證指令。

        回傳 ``(驗證輸出, is_healthy)``；若沒有匹配的驗證規則回傳 None。
        對於 docker 類操作，會優先使用 ``docker compose ps``（以子任務
        裡的 ``cd X`` 或 ``-f path`` 當作專案根），只看本次任務啟動的
        服務，而不是整台主機的所有容器。在偵測到 unhealthy / Restarting /
        Exited 時自動抓取 ``docker logs`` 與 ``docker inspect``，並把
        is_healthy 設為 False 以觸發升級。
        """
        desc = st.description
        desc_lower = desc.lower()

        # Docker 操作：嘗試用 scoped 指令避免看到無關容器
        docker_keywords = ["docker compose", "docker-compose", "docker run", "docker"]
        if any(kw in desc_lower for kw in docker_keywords):
            # Priority 1: explicit `-f path` or `cd X` in the description
            scoped_cmd = self._compose_scoped_ps_cmd(desc)
            # Priority 2: fall back to the session's effective workspace.
            # This stops verification from doing `docker ps` globally,
            # which would pick up unrelated unhealthy containers on the
            # host (e.g. mold-harvester-db-1 leftover from a previous
            # project) and trigger bogus Claude escalations.
            if not scoped_cmd:
                try:
                    from config import effective_workspace_dir
                    import shlex as _shlex
                    ws = effective_workspace_dir()
                    if ws:
                        scoped_cmd = (
                            f"cd {_shlex.quote(ws)} && "
                            "docker compose ps --format "
                            "'table {{.Name}}\\t{{.Status}}\\t{{.Service}}' "
                            "2>/dev/null || docker ps"
                        )
                except Exception:  # noqa: BLE001
                    pass
            # Priority 3: global docker ps (last resort)
            if not scoped_cmd:
                scoped_cmd = "docker ps"
            logger.info("Running verification '%s' for subtask %s", scoped_cmd, st.id)
            output = await self._run_cmd_capture(scoped_cmd, timeout=10.0, max_chars=1200)
            if output is None:
                return None
            return await self._inspect_docker_health(output)

        # 其他類型：固定映射
        checks: List[Tuple[List[str], str]] = [
            (["git clone"],                    "ls -la"),
            (["pip install"],                  "pip list | head -20"),
            (["npm install", "yarn install"],  "ls node_modules 2>/dev/null | head -10 || echo 'node_modules not found'"),
            (["mkdir"],                        "ls -la"),
            (["cp ", "copy "],                 "ls -la"),
        ]
        for keywords, cmd in checks:
            if any(kw in desc_lower for kw in keywords):
                logger.info("Running verification '%s' for subtask %s", cmd, st.id)
                output = await self._run_cmd_capture(cmd, timeout=10.0, max_chars=800)
                if output is None:
                    return None
                return (output, True)

        return None  # 沒有匹配的驗證規則

    @staticmethod
    def _compose_scoped_ps_cmd(desc: str) -> Optional[str]:
        """Produce a scoped ``docker compose ps`` command when the subtask
        clearly targets a specific compose project.

        Picks the first of these signals that appears in the description:
        - ``-f <path>`` / ``--file <path>``  → ``docker compose -f <path> ps``
        - ``cd <dir>``                        → ``cd <dir> && docker compose ps``

        Returns None when no target can be extracted, letting the caller
        fall back to plain ``docker ps``.
        """
        f_match = re.search(r"(?:^|\s)(?:-f|--file)(?:\s+|=)(\S+)", desc)
        if f_match:
            path = f_match.group(1).strip().rstrip(";&|")
            return (
                f"docker compose -f {path} ps --format "
                "'table {{.Name}}\\t{{.Status}}\\t{{.Service}}'"
            )
        cd_match = re.search(r"\bcd\s+([^\s&|;<>]+)", desc)
        if cd_match:
            path = cd_match.group(1).strip()
            return (
                f"cd {path} && docker compose ps --format "
                "'table {{.Name}}\\t{{.Status}}\\t{{.Service}}'"
            )
        return None

    async def _inspect_docker_health(self, ps_output: str) -> Tuple[str, bool]:
        """Scan ``docker ps`` output. If any row shows unhealthy/Restarting/
        Exited, pull logs + inspect and return is_healthy=False."""
        bad_names: List[str] = []
        for line in ps_output.splitlines()[1:]:  # skip header
            if not line.strip():
                continue
            if _DOCKER_UNHEALTHY_RE.search(line):
                parts = line.split()
                if parts:
                    bad_names.append(parts[-1])  # container name is last column

        if not bad_names:
            return (ps_output, True)

        # Collect logs + health state for up to 3 unhealthy containers
        sections: List[str] = [ps_output, ""]
        for name in bad_names[:3]:
            sections.append(f"--- {name} logs (tail=100) ---")
            logs = await self._run_cmd_capture(
                f"docker logs --tail=100 --timestamps {name} 2>&1",
                timeout=15.0,
                max_chars=1500,
            )
            sections.append(logs or "(no log output)")
            sections.append(f"\n--- {name} health state ---")
            health = await self._run_cmd_capture(
                f"docker inspect --format '{{{{json .State.Health}}}}' {name}",
                timeout=5.0,
                max_chars=800,
            )
            sections.append(health or "(no health state)")
            sections.append("")

        return ("\n".join(sections), False)

    @staticmethod
    async def _run_cmd_capture(
        cmd: str, timeout: float = 10.0, max_chars: int = 800,
    ) -> Optional[str]:
        """Run a shell command and return its stdout/stderr, or None on failure."""
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            out = stdout_b.decode(errors="replace").strip()
            err = stderr_b.decode(errors="replace").strip()
            return (out or err or "(no output)")[:max_chars]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Command '%s' failed (non-fatal): %s", cmd[:80], exc)
            return None

    async def _gemma_summarize(self, goal: str, tool_log: str, task_id: str = "") -> str:
        """用 Gemma 讀取工具執行記錄，產出最後要交回給使用者的回應。

        關鍵設計：分辨「成品任務」vs「動作任務」。
          - 動作任務（部署 / 安裝 / 設定 / 啟動）→ 30-200 字簡潔簡報。
          - 成品任務（產出列表 / 整理表格 / 生成 / 寫 / 翻譯 / 提取）
            → 直接把工具產出的內容**原封不動**交回，不寫摘要。

        歷史 bug：以前 prompt 把每個任務都當「動作任務」寫簡報。
        「產出 GitHub trending top 10」這種任務，python_exec 明明已經
        印出 markdown 列表，最後使用者卻只看到「已完成提取與格式化」
        的摘要，列表被吃掉。
        """
        system_prompt = """\
你是任務結果產生器。根據工具執行記錄，產出最後要交回給使用者的內容。

【核心分辨：成品任務 vs 動作任務】

▸ 成品任務（使用者要的就是內容本身）
  關鍵動詞：產出 / 生成 / 列出 / 整理 / 寫 / 編輯 / 翻譯 / 摘要 /
            提取 / 抓取（並輸出）/ produce / generate / list /
            extract / write / draft / summarize / translate
  → 把工具產出的實質內容（markdown 列表、表格、JSON、程式碼、
     文字）原封不動貼上來。不要寫「已完成」，不要寫「成功解析」。
     使用者要的就是那段內容本身。
  → 若 python_exec / write_file 印出了結構化內容，那段就是成品。

▸ 動作任務（使用者要做一件事，結果是 side effect）
  關鍵動詞：部署 / 安裝 / 修復 / 設定 / 啟動 / 確認 / 跑 / deploy /
            install / fix / configure / run / start
  → 寫一段簡潔的繁體中文簡報（30-200 字），描述做了什麼、結果如何。

【判斷流程】
  1. 看使用者的 goal 動詞 → 推斷是「成品」還是「動作」。
  2. 看工具記錄裡有沒有「結構化的實質產出」
     （markdown 列表 / 表格 / JSON / 程式碼 / 大段文字）。
  3. 兩者吻合 → 那段產出就是答案，原樣返回。
  4. 若是動作任務 → 才寫摘要報告。
  5. 不確定時優先「原樣返回」，不要硬寫摘要。

【成品任務範例】
  Goal: 產出 GitHub Trending 今日前 10 名專案的 markdown 列表
  python_exec 印出：
    - **microsoft/markitdown** — Lightweight tool for converting... (Python, 2,341 stars today)
    - **vllm-project/vllm** — A high-throughput inference engine... (Python, 1,852 stars today)
    - ...
  ✅ 正確回應：直接把上面那 10 條 markdown 貼回，一字不改。
  ❌ 錯誤回應：「已完成 GitHub Trending 前 10 名熱門專案的提取與格式化」

【動作任務範例】
  Goal: 部署 Q1 銷售分析報告到 nginx
  ✅ 正確回應：
    完成 Q1 銷售分析報告部署。
    - 容器 sales-nginx 已啟動，listen :8080
    - 3 張圖表渲染完成
    📎 產出：reports/q1_analysis.html

【規則】
- 絕不要寫「我是 AI 我不能...」—— 只描述實際發生的事。
- 只能描述記錄裡真實存在的內容；不相關的忽略。
- 成品任務優先：使用者明確要產出 X，就交付 X 的內容本身。
- 不確定就傾向把 python_exec 最後一段結構化 stdout 原樣返回。
- 動作任務才用繁體中文簡報；成品任務跟隨使用者原本的語言。
- 若 render_chart_report / write_file 有產出檔案，動作摘要末尾
  加一行「📎 產出：<relative/path.html>」。
"""
        prompt = (
            f"用戶的目標：{goal}\n\n"
            f"工具執行記錄：\n{tool_log}\n\n"
            "請根據以上記錄生成完成報告。"
        )
        # Prepend current-time header so synthesizer doesn't default to
        # its training-cutoff year if the goal asks anything time-related
        # (calendar, "what year is it", lunar date, etc).
        system_prompt = _current_time_header() + system_prompt
        # Same cloud-aware routing as planning — see _orchestrator_complete.
        result = await self._orchestrator_complete(
            prompt, system_prompt,
            task_id=task_id,
            task_description=f"[summary] {goal[:100]}",
        )
        # Fallback if Gemma / cloud returns empty. We dump a generous slice
        # of the tool log so a markdown list / table / JSON deliverable
        # doesn't get sliced in half mid-row — better to show the user
        # 8 KB of raw output than a half-rendered list.
        if not result or result.strip() in ("[]", ""):
            lines = [f"{_L(goal, 'task_completed')}: {goal}", "", tool_log[:8000]]
            return "\n".join(lines)
        return result

    # ── Intent classifier (fast-path for pure-chat goals) ────────────────────
    # Run at the top of orchestrator.run() — classifies the user's goal as
    # "needs tools / pure chat" via a single Gemma call (~100ms on a local
    # model, longer on cloud). When the classifier is confident and says
    # the goal doesn't need tools, the run() loop short-circuits to a
    # single-turn _fast_chat_response(), saving 5-30s of empty subtask /
    # tool loop work on every "what is X" / "explain Y" / "how does Z"
    # question. Conservative — when in doubt, fall back to full flow.
    _INTENT_CLASSIFIER_SYSTEM = (
        "You classify a user goal into routing decisions for an AI agent. "
        "Reply with ONE JSON object and nothing else (no markdown fences, "
        "no prose):\n"
        '{"needs_tools": <bool>, "is_pure_chat": <bool>, '
        '"confidence": <0.0-1.0>, "rationale": "<short>"}\n\n'
        "needs_tools=true  → goal genuinely needs shell / file / db / http / "
        "code execution (build something, install, deploy, analyse a file, "
        "query a DB, scrape a URL, …).\n"
        "needs_tools=false → goal is answerable from training data + "
        "memory alone (explain, describe, define, compare concepts, recommend, "
        "what / how / why / which questions about general knowledge).\n"
        "is_pure_chat=true → conversational / informational; output is "
        "going to be prose, not files or side effects.\n"
        "confidence: 0.5-1.0. Below 0.7 means \"not sure\" — caller will "
        "default to the full tool loop rather than risk skipping needed work.\n\n"
        "Examples:\n"
        '- "Build a Next.js GitHub trending app" → {"needs_tools": true, "is_pure_chat": false, "confidence": 0.95, "rationale": "build implies file creation"}\n'
        '- "What is OpenTeddy?" → {"needs_tools": false, "is_pure_chat": true, "confidence": 0.95, "rationale": "pure info question"}\n'
        '- "Explain how docker compose works" → {"needs_tools": false, "is_pure_chat": true, "confidence": 0.92, "rationale": "explanation only"}\n'
        '- "Analyze sales.csv" → {"needs_tools": true, "is_pure_chat": false, "confidence": 0.9, "rationale": "needs file access"}\n'
        '- "What\'s a good model for code generation?" → {"needs_tools": false, "is_pure_chat": true, "confidence": 0.9, "rationale": "opinion / recommendation"}\n'
        '- "Install pandas" → {"needs_tools": true, "is_pure_chat": false, "confidence": 0.9, "rationale": "pip install side effect"}\n'
        '- "Tell me what frameworks this app uses" → {"needs_tools": true, "is_pure_chat": false, "confidence": 0.8, "rationale": "needs to read project files"}'
    )

    async def _classify_intent(
        self, goal: str, session_mode: str,
    ) -> Optional[Dict[str, Any]]:
        """Single-shot intent classification. Returns a dict with
        needs_tools, is_pure_chat, confidence, rationale — or None on
        any parse / network failure (caller defaults to full flow).

        Uses _orchestrator_complete so the call routes through the
        currently-active provider (local Gemma in mixed / local mode,
        cloud LLM in cloud mode) without separate plumbing.
        """
        if not goal or not goal.strip():
            return None
        prompt = f"Session mode hint: {session_mode}\nUser goal: {goal.strip()}"
        try:
            raw = await self._orchestrator_complete(
                prompt,
                self._INTENT_CLASSIFIER_SYSTEM,
                task_description="[intent-classify]",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("intent classify LLM call failed: %s", exc)
            return None
        # Be tolerant of extra prose / code fences — small models
        # sometimes can't help themselves even with explicit instructions.
        import re as _re
        match = _re.search(r"\{[\s\S]*\}", raw or "")
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        return {
            # Safe defaults — when a key is missing, prefer "do the full
            # work" over "skip tools".
            "needs_tools":   bool(data.get("needs_tools", True)),
            "is_pure_chat":  bool(data.get("is_pure_chat", False)),
            "confidence":    float(data.get("confidence", 0.5) or 0.5),
            "rationale":     str(data.get("rationale", ""))[:200],
        }

    async def _fast_chat_response(
        self,
        req: TaskRequest,
        session_mode: str,
        intent: Dict[str, Any],
    ) -> TaskResult:
        """Pure-chat fast path: bypass plan + execute + summary entirely,
        answer the goal in a single LLM turn. Used when _classify_intent
        flagged the goal as needs_tools=False with high confidence.

        We still:
          - Persist a synthetic single-SubTask for tracker schema
            integrity (DB queries assume every task has subtasks)
          - Write to long-term memory so the conversation thread is
            preserved
          - Fire the skill auto-detection check (a recurring chat
            question is just as valid a skill candidate as a recurring
            tool task)
        """
        # Pull memory context the same way _plan does so the answer
        # benefits from session history.
        memory_ctx = ""
        if self.memory is not None and self.memory.is_available:
            try:
                memory_ctx = await self.memory.get_context_for_task(
                    req.goal, session_id=req.session_id,
                )
            except Exception:  # noqa: BLE001
                pass

        # Current-time header FIRST so a "today / 今天 / 現在 / what year"
        # question answered on this fast path doesn't fall back to the
        # model's training-cutoff year. This path skips the planner +
        # synthesizer (which already carry the header), so without this
        # it's the one place a date question gets the wrong year — the
        # exact "今天農曆幾月幾號 → 2024" bug.
        system = (
            _current_time_header()
            + "You are OpenTeddy, a helpful and concise AI assistant. "
            "The user asked a question that doesn't require running "
            "tools — answer directly. Keep the response tight and "
            "useful; use markdown for structure where it helps."
        )
        if memory_ctx:
            system += "\n\n--- Relevant memory ---\n" + memory_ctx

        # Stream so the user gets tokens as they generate — same UX as
        # the full-loop chat-mode finalize step.
        answer = await self._orchestrator_complete(
            req.goal,
            system,
            task_id=req.id,
            task_description=f"[fast-chat] {req.goal[:100]}",
        )
        if not answer or not answer.strip():
            # Defensive: empty response shouldn't happen but let the
            # full flow handle it rather than returning blank.
            logger.warning(
                "Fast-path returned empty answer — falling back to full flow"
            )
            raise RuntimeError("fast-path empty response")

        # Synthetic subtask record. Confidence carries the classifier's
        # confidence so the Usage tab perf stats can distinguish fast-
        # path tasks from full-loop ones.
        fake_st = SubTask(
            parent_task_id=req.id,
            description=req.goal,
            agent=AgentRole.EXECUTOR,
            order=0,
            status=TaskStatus.COMPLETED,
            result=answer.strip(),
            confidence=float(intent.get("confidence", 0.9)),
        )
        try:
            await self.tracker.create_subtask(fake_st)
            await self.tracker.update_subtask(fake_st)
            await self.tracker.update_task_status(
                req.id, TaskStatus.COMPLETED, answer.strip(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fast-path tracker write failed: %s", exc)

        # Memory write — same as full flow's step 5.
        if self.memory is not None:
            try:
                await self.memory.summarize_and_store(
                    task_id=req.id,
                    goal=req.goal,
                    final_output=answer,
                    session_id=req.session_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Memory store (fast-path) failed: %s", exc)

        # Skill detection still fires — a recurring "what is X" pattern
        # is a legitimate skill candidate (could become a cached info
        # response).
        if (self.memory is not None
                and self.memory.is_available
                and not _is_local_only()):
            asyncio.create_task(
                self._maybe_promote_pattern_to_skill(req)
            )

        return TaskResult(
            task_id=req.id,
            status=TaskStatus.COMPLETED,
            summary=answer.strip(),
            subtasks=[fake_st],
            skills_used=[],
            new_skills_created=[],
        )

    # ── Skill auto-detection (embedding-based) ────────────────────────────────
    # Why this exists: the original mechanism required Qwen to emit
    # skill_needed + skill_description in its JSON output, but 2-3B
    # parameter models almost never do metacognitive "this could be a
    # reusable function" reflection — and we have empirical confirmation
    # that 0 skills have been generated in 1+ month of real usage.
    #
    # New approach: AFTER each successful task, run a semantic similarity
    # search against past task_result memories. If ≥ N past goals match
    # the current goal at ≥ similarity threshold, that's a recurring
    # pattern → synthesize a skill name + description via the cloud LLM
    # and ask skill_factory to materialise it.
    #
    # Background-fires from orchestrator.run() so the user gets their
    # task result back without waiting for the embedding lookup + LLM
    # synthesis (1-3 s total).
    async def _maybe_promote_pattern_to_skill(
        self, req: TaskRequest,
    ) -> None:
        """Detect recurring-goal patterns and ask SkillFactory to make one.

        No-op on missing memory, disabled detection, or insufficient
        repetition. Swallows all errors — skill auto-detection is a
        nice-to-have, not a correctness guarantee.
        """
        try:
            from config import config as _cfg
            min_repeats = int(getattr(_cfg, "skill_auto_detect_min_repeats", 3))
            similarity_threshold = float(
                getattr(_cfg, "skill_auto_detect_similarity", 0.85)
            )
            if min_repeats <= 0:
                return  # explicitly disabled

            # Search across ALL sessions (no session_id filter) — recurring
            # patterns can appear in different sessions and that's exactly
            # the case where a skill is most useful.
            similar = await self.memory.search_memory(
                query=req.goal,
                n_results=min_repeats + 5,
            )

            # Keep only task_result memories from OTHER tasks above threshold.
            # Excluding self.id matters because this task's own memory has
            # already been written (step 5 fires before us in run()).
            matching = [
                m for m in (similar or [])
                if m.get("type") == "task_result"
                   and m.get("task_id") != req.id
                   and float(m.get("relevance_score", 0.0)) >= similarity_threshold
            ]
            if len(matching) < min_repeats:
                logger.debug(
                    "Skill auto-detect: only %d similar tasks (need %d) — "
                    "not promoting yet",
                    len(matching), min_repeats,
                )
                return

            # Pattern confirmed — synthesise a name + description.
            cluster_goals = [req.goal] + [
                str(m.get("content", ""))[:200] for m in matching
            ]
            skill_name, skill_desc = await self._synthesize_skill_from_cluster(
                cluster_goals, task_id=req.id,
            )
            if not skill_name or not skill_desc:
                logger.info(
                    "Skill auto-detect: pattern matched (%d×) but synthesis "
                    "returned empty name/desc — skipping", len(matching),
                )
                return

            # De-dup: only ask SkillFactory once per name.
            try:
                existing = await self.skill_factory.tracker.get_skill(skill_name)
                if existing:
                    logger.debug(
                        "Skill auto-detect: '%s' already exists — skipping",
                        skill_name,
                    )
                    return
            except Exception:  # noqa: BLE001
                pass  # tracker lookup is best-effort; proceed

            logger.info(
                "Skill auto-detect: recurring pattern (%d matches ≥ %.2f) → "
                "generating skill '%s'",
                len(matching), similarity_threshold, skill_name,
            )
            await self.skill_factory.generate_skill(skill_name, skill_desc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skill auto-detect failed (non-fatal): %s", exc,
            )

    async def _synthesize_skill_from_cluster(
        self, cluster_goals: List[str], task_id: str = "",
    ) -> Tuple[Optional[str], Optional[str]]:
        """Ask the orchestrator LLM to derive a skill name + description
        from a cluster of similar goals. Returns (name, description) or
        (None, None) on any failure / malformed response.

        Uses _orchestrator_complete so the call respects llm_mode (cloud
        in cloud mode, local Gemma otherwise) — no extra config plumbing.
        """
        if not cluster_goals:
            return None, None

        # Compact prompt — small models do better with sharp constraints.
        # The cluster is small (3-10 goals) so we don't need to truncate.
        goals_blob = "\n".join(
            f"- {g.strip()[:240]}" for g in cluster_goals[:10] if g
        )
        system = (
            "You analyse a CLUSTER of similar user goals from an AI "
            "agent's history and decide if they share a clear reusable "
            "pattern. Reply ONLY with a single JSON object in this exact "
            "shape — no markdown fences, no preamble:\n"
            '{"name": "<snake_case, max 40 chars>", '
            '"description": "<single sentence under 200 chars describing '
            'what the reusable function does>"}\n'
            "If the goals are too heterogeneous to form one skill, reply "
            'with {"name": "", "description": ""} instead.'
        )
        prompt = (
            "Cluster of recurring goals (most recent first):\n"
            f"{goals_blob}\n\n"
            "What is the single reusable skill that captures this pattern?"
        )

        try:
            raw = await self._orchestrator_complete(
                prompt, system,
                task_id=task_id,
                task_description="[skill-synth]",
            )
            # Best-effort JSON parse — tolerant of extra prose / fences
            # the model sometimes wraps despite the instruction.
            import re as _re
            match = _re.search(r"\{[\s\S]*\}", raw or "")
            if not match:
                return None, None
            data = json.loads(match.group())
            name = (data.get("name") or "").strip()
            desc = (data.get("description") or "").strip()
            if not name or not desc:
                return None, None
            # Sanitise name: snake_case, alphanumeric + underscore only,
            # ≤ 40 chars. Falls back to the first 40 chars stripped of
            # junk if the model gave us something weird.
            import re as _re2
            name = _re2.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:40]
            if not name:
                return None, None
            return name, desc[:240]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skill synthesis failed: %s", exc)
            return None, None

    async def _run_confirmation_checks(
        self, goal: str, subtasks: Optional[List[SubTask]] = None,
        session_mode: str = "code",
    ) -> str:
        """根據任務目標關鍵字，執行確認指令並回傳格式化結果。

        對於 Docker 任務，會從子任務描述中擷取 compose 路徑（``cd X`` 或
        ``-f path``），優先使用 ``docker compose ps`` 只看本次任務的服務，
        不汙染主機上其他既有容器。擷取不到才退回全域 ``docker ps``。

        Gated by session mode:
          - **Code** mode: runs the block (deploys, installs, git ops
            all genuinely benefit from a "did it actually come up?"
            confirmation at the end).
          - **Chat / Analytic**: skipped. Analytic produces markdown or
            a chart report — dumping ``ls -la ~/`` after an analysis
            summary is pure noise (the user explicitly complained).

        Keyword matching also uses word boundaries for short tokens
        (``git``, ``repo``, ``npm``, ``node``) so that a goal like
        "generate an analysis **report**" no longer matches "repo" and
        triggers an irrelevant home-directory listing.
        """
        # Skip entirely for non-code modes.
        if session_mode not in ("code", "coding"):
            return ""

        import re as _re

        def _has_word(text: str, word: str) -> bool:
            """Substring match for multi-char tokens, word-boundary match
            for short ones where a substring match would produce false
            positives (repo → report, npm → nympo, node → anode).
            """
            if len(word) <= 4:
                return bool(_re.search(rf"\b{_re.escape(word)}\b", text))
            return word in text

        goal_lower = goal.lower()
        confirmation_cmds: List[Tuple[str, str]] = []

        if _has_word(goal_lower, "docker"):
            scoped_cmd = None
            for st in (subtasks or []):
                scoped_cmd = self._compose_scoped_ps_cmd(st.description or "")
                if scoped_cmd:
                    break
            if scoped_cmd:
                confirmation_cmds.append((_L(goal, "container_status"), scoped_cmd))
            else:
                confirmation_cmds.append((_L(goal, "docker_status"), "docker ps"))
        if any(_has_word(goal_lower, kw) for kw in ["clone", "git", "repository", "repo"]):
            # Use the effective workspace, not `~/` — listing the user's
            # whole home dir was leaking personal files into the summary.
            from config import effective_workspace_dir
            ws = effective_workspace_dir()
            confirmation_cmds.append(
                (_L(goal, "workspace_contents"), f"ls -la {ws!r} 2>/dev/null || ls -la .")
            )
        if any(_has_word(goal_lower, kw) for kw in ["install", "pip", "package"]):
            confirmation_cmds.append((_L(goal, "installed_pkgs"), "pip list 2>/dev/null | head -20"))
        if any(_has_word(goal_lower, kw) for kw in ["npm", "node", "yarn"]):
            confirmation_cmds.append((_L(goal, "node_packages"), "ls node_modules 2>/dev/null | head -10 || echo 'node_modules not found'"))

        if not confirmation_cmds:
            return ""

        output_parts: List[str] = [_L(goal, "confirmation_header")]
        for label, cmd in confirmation_cmds:
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=15.0
                )
                out = stdout_b.decode(errors="replace").strip()
                err = stderr_b.decode(errors="replace").strip()
                result_text = (out or err or "(無輸出)")[:500]
                output_parts.append(f"\n**{label}**\n```\n{result_text}\n```")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Confirmation check '%s' failed (non-fatal): %s", cmd, exc)

        return "\n".join(output_parts) if len(output_parts) > 1 else ""

    async def _orchestrator_complete(
        self, prompt: str, system: Optional[str] = None,
        task_id: str = "", task_description: str = "[orchestrator]",
    ) -> str:
        """Orchestrator-side text completion that respects llm_mode.

        Cloud mode → route to the same provider escalation uses, so the
        user truly doesn't need Ollama installed. Local / Mixed → keep
        the existing Gemma-via-Ollama path. On cloud failure (bad key,
        rate-limit, network blip) we fall back to Gemma so a transient
        cloud issue doesn't brick planning entirely — the user can see
        the warning in logs and fix it without losing the task.
        """
        from config import is_cloud_mode
        if not is_cloud_mode():
            return await self._gemma_complete(
                prompt, system, task_id=task_id, task_description=task_description,
            )

        provider = self.escalation.provider
        import time as _time
        start = _time.monotonic()
        try:
            resp = await provider.complete_text(
                user_message=prompt,
                system=system or _PLAN_SYSTEM_BASE,
                max_tokens=int(getattr(config, "gemma_max_tokens", 4096) or 4096),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cloud-mode orchestrator call failed (%s) — falling back to "
                "Gemma. Check Settings → Cloud LLM Provider key.", exc,
            )
            return await self._gemma_complete(
                prompt, system, task_id=task_id, task_description=task_description,
            )

        duration_ms = int((_time.monotonic() - start) * 1000)

        # Mirror the Gemma path's usage record so the Usage tab + cost
        # accounting still see this call. Best-effort: tracker failures
        # don't break planning.
        try:
            await self.tracker.record_usage(
                task_id=task_id,
                model=provider.model_name,
                model_provider=provider.provider_name,
                tokens_in=resp.usage.input_tokens,
                tokens_out=resp.usage.output_tokens,
                task_description=task_description,
                duration_ms=duration_ms,
            )
        except Exception:  # noqa: BLE001
            pass

        return resp.text

    async def _gemma_complete(
        self, prompt: str, system: Optional[str] = None,
        task_id: str = "", task_description: str = "[orchestrator]",
    ) -> str:
        # Stream if globally enabled — this is #3 (Streaming for
        # orchestrator). Without it the user stares at a blank "Working
        # on it… 0:14" for 5–15 s while Gemma plans the task. With it
        # enabled, JSON / reasoning text appears chunk-by-chunk so the
        # user immediately sees the agent is alive and thinking.
        stream_on = bool(getattr(config, "streaming_enabled", True))
        # Reach into the executor for its ws_callback so we don't have
        # to thread one separately into the orchestrator constructor.
        ws_emit = getattr(self.executor, "ws_callback", None) if stream_on else None

        payload = {
            "model":   config.gemma_model,
            "prompt":  prompt,
            "system":  system or _PLAN_SYSTEM_BASE,
            "stream":  stream_on,
            "options": {
                "temperature": float(getattr(config, "gemma_temperature", 0.1)),
                "num_predict": config.gemma_max_tokens,
                "num_ctx":     int(getattr(config, "gemma_num_ctx", 16384)),
            },
            # Same per-request keep_alive override the executor uses —
            # so orchestrator's plan + fast-chat / classifier calls all
            # benefit from the long retention setting without touching
            # Ollama's daemon config.
            "keep_alive": getattr(config, "ollama_keep_alive", "24h"),
        }
        try:
            if stream_on:
                # /api/generate streams as NDJSON, each line:
                #   {"response": "...", "done": false}
                # final line carries done: true plus prompt/eval counts.
                acc = ""
                last: Dict[str, Any] = {}
                async with self._http.stream(
                    "POST",
                    f"{config.gemma_base_url}/api/generate",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        delta = chunk.get("response") or ""
                        if delta:
                            acc += delta
                            if ws_emit:
                                try:
                                    await ws_emit({
                                        "type":    "plan.stream.delta",
                                        "task_id": task_id,
                                        "text":    delta,
                                    })
                                except Exception:  # noqa: BLE001
                                    pass
                        if chunk.get("done"):
                            last = chunk
                # Mark stream end so the UI can switch from "planning…"
                # placeholder to the regular "Working on it…" timer.
                if ws_emit:
                    try:
                        await ws_emit({
                            "type":    "plan.stream.end",
                            "task_id": task_id,
                        })
                    except Exception:  # noqa: BLE001
                        pass
                tokens_in    = last.get("prompt_eval_count", 0) or 0
                tokens_out   = last.get("eval_count", 0) or 0
                eval_dur_ns  = last.get("eval_duration", 0) or 0
                total_dur_ns = last.get("total_duration", 0) or 0
                response_text = acc
            else:
                resp = await self._http.post(
                    f"{config.gemma_base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                tokens_in    = data.get("prompt_eval_count", 0) or 0
                tokens_out   = data.get("eval_count", 0) or 0
                eval_dur_ns  = data.get("eval_duration", 0) or 0
                total_dur_ns = data.get("total_duration", 0) or 0
                response_text = data.get("response", "")

            # ── Record Ollama usage (best-effort) ─────────────────────────────
            # #6 Auto-benchmark: include real wall-clock + measured t/s
            # so the Settings perf-stats query has data to aggregate.
            if tokens_in or tokens_out:
                tps = (
                    tokens_out * 1_000_000_000 / eval_dur_ns
                    if eval_dur_ns > 0 and tokens_out > 0 else 0.0
                )
                try:
                    await self.tracker.record_usage(
                        task_id=task_id,
                        model=config.gemma_model,
                        model_provider="ollama",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        task_description=task_description,
                        duration_ms=int(total_dur_ns / 1_000_000),
                        tokens_per_sec=tps,
                    )
                except Exception:  # noqa: BLE001
                    pass

            return response_text
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemma call failed: %s", exc)
            return "[]"

    @staticmethod
    def _parse_plan(raw: str, task_id: str, goal: str = "") -> List[SubTask]:
        """Parse Gemma's plan output into SubTask list.

        Tries multiple strategies to handle truncated or malformed JSON output:
        1. Direct JSON parse of the full response
        2. Slice from first '[' to last ']' (handles leading/trailing prose)
        3. Greedy regex match for a JSON array (handles code fences etc.)
        4. Extract individual JSON objects even from a truncated array
        5. Fallback: single subtask from the original goal so Qwen always runs
        """

        def _build_subtasks(items: list) -> List[SubTask]:
            subtasks: List[SubTask] = []
            for item in items[:10]:
                if not isinstance(item, dict):
                    continue
                desc = item.get("description", "").strip()
                if not desc:
                    continue
                subtasks.append(
                    SubTask(
                        parent_task_id=task_id,
                        description=desc,
                        skill_hint=item.get("skill_hint") or None,
                        agent=AgentRole.EXECUTOR,
                        order=int(item.get("order", len(subtasks))),
                    )
                )
            subtasks.sort(key=lambda s: s.order)
            return subtasks

        # ── Strategy 1: direct parse ─────────────────────────────────────────
        try:
            items = json.loads(raw.strip())
            if isinstance(items, list):
                result = _build_subtasks(items)
                if result:
                    return result
        except json.JSONDecodeError:
            pass

        # ── Strategy 2: first '[' … last ']' ────────────────────────────────
        start = raw.find("[")
        end   = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                items = json.loads(raw[start : end + 1])
                if isinstance(items, list):
                    result = _build_subtasks(items)
                    if result:
                        return result
            except json.JSONDecodeError:
                pass

        # ── Strategy 3: greedy regex (handles markdown fences) ───────────────
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
                if isinstance(items, list):
                    result = _build_subtasks(items)
                    if result:
                        return result
            except json.JSONDecodeError:
                pass

        # ── Strategy 4: extract individual objects from truncated array ───────
        obj_matches = re.findall(r'\{[^{}]*"description"[^{}]*\}', raw, re.DOTALL)
        if obj_matches:
            items = []
            for i, obj_str in enumerate(obj_matches):
                try:
                    obj = json.loads(obj_str)
                    if "order" not in obj:
                        obj["order"] = i
                    items.append(obj)
                except json.JSONDecodeError:
                    pass
            if items:
                result = _build_subtasks(items)
                if result:
                    logger.warning(
                        "Parsed %d subtasks from truncated JSON via object extraction.",
                        len(result),
                    )
                    return result

        # ── Strategy 5: fallback — at least let Qwen run something ───────────
        # The subtask description is passed verbatim into the executor's
        # user message. Earlier this was prefixed with "執行用戶的原始請求:"
        # which read to small chat-mode models as a verb instruction
        # ("execute the original request") and made them produce
        # past-tense action reports — e.g. user says "早安", executor
        # replies "已完成用戶的問候回應 / 已回覆用戶問候" instead of just
        # saying "早安！👋". Pass `goal` raw so the model treats it as
        # the actual question to answer.
        logger.warning(
            "Could not parse Gemma plan JSON (raw length=%d); "
            "falling back to single subtask from goal.",
            len(raw),
        )
        if goal:
            return [
                SubTask(
                    parent_task_id=task_id,
                    description=goal,
                    skill_hint=None,
                    agent=AgentRole.EXECUTOR,
                    order=0,
                )
            ]
        return []

    @staticmethod
    def _derive_status(subtasks: List[SubTask]) -> TaskStatus:
        statuses = {st.status for st in subtasks}
        if TaskStatus.FAILED in statuses:
            if all(s in (TaskStatus.FAILED,) for s in statuses):
                return TaskStatus.FAILED
            return TaskStatus.COMPLETED  # partial success
        if TaskStatus.ESCALATED in statuses:
            return TaskStatus.COMPLETED
        return TaskStatus.COMPLETED

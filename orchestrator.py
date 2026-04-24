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
並產生**漂亮的報告 + 圖表**。圖表在前端用 Chart.js 渲染。

【標準分析流程 — 請照這個拆】

1. **探查資料** — 用 shell_exec_readonly：
   - `ls -la uploads/` 看有什麼檔
   - `head -5 uploads/<file>.csv` 或 `python3 -c "import pandas as pd; print(pd.read_csv('uploads/<file>.csv').head())"` 看欄位
   - `wc -l uploads/<file>.csv` 看行數

2. **執行分析** — 用 shell_exec_write 跑 Python（pandas / numpy）：
   - 計算統計量、分組彙總、時序趨勢、相關性等
   - 把結果存成中間檔（/tmp/analysis.json）方便下一步讀

3. **產出報告** — 最後一步請**明確指示** executor 產出：
   - markdown 格式的報告（標題、段落、重點列表）
   - 2-5 個 ```chart JSON 區塊，每個代表一張 Chart.js 圖表
   - 區塊格式（Chart.js v4）：
     ```
     ```chart
     {
       "type": "bar" | "line" | "pie" | "doughnut" | "scatter" | "radar",
       "data": {
         "labels": ["Jan", "Feb", "Mar"],
         "datasets": [{
           "label": "Revenue",
           "data": [100, 150, 130],
           "backgroundColor": "#d97757"
         }]
       },
       "options": {
         "plugins": {"title": {"display": true, "text": "Q1 Revenue"}}
       }
     }
     ```
     ```
   - 前端會偵測這些 ```chart 區塊並即時渲染成互動式圖表

【規則】
- 跟 Code 模式一樣：具體指令、cd 用 && 串、禁止「請使用者執行」
- 檔案路徑一律**相對於 workspace**（例如 `uploads/sales.csv`），
  不要用 `../` 或絕對路徑跳出去
- 子任務數量 3-5 個（探查 / 分析 / 產出報告三步是最常見）
- 最後一步**必須**是「把結果整理成 markdown 報告，包含 N 張 Chart.js 圖」

【常用的圖表選擇】
- 時序趨勢 → line
- 類別比較 → bar
- 佔比 → pie / doughnut
- 關聯性 → scatter
- 多維度比較 → radar

輸出純 JSON 陣列，只輸出 JSON，不要任何其他文字、markdown、說明。
每個元素包含：
  - "description": string  (具體操作描述)
  - "skill_hint": string or null
  - "order": integer
"""


# Back-compat alias — some older code imports _PLAN_SYSTEM_BASE directly.
_PLAN_SYSTEM_BASE = _PLAN_SYSTEM_CODE


def _plan_prompt_for_mode(mode: str) -> str:
    """Pick the plan-system prompt that matches the session's mode."""
    if mode == "chat":     return _PLAN_SYSTEM_CHAT
    if mode == "analytic": return _PLAN_SYSTEM_ANALYTIC
    return _PLAN_SYSTEM_CODE   # default — safest for unknown modes


def _is_local_only() -> bool:
    """Thin wrapper so orchestrator code reads cleanly at call sites."""
    from config import is_session_local_only
    return is_session_local_only()


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
        if req.session_id:
            try:
                sess = await self.tracker.get_session(req.session_id)
                if sess:
                    session_ws = sess.get("workspace_dir") or None
                    session_local_only = bool(sess.get("local_only"))
            except Exception:  # noqa: BLE001
                pass
        set_session_workspace(session_ws)
        set_session_local_only(session_local_only)
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
            for st in subtasks:
                st = await self._run_subtask(st, req.context, mode=mode_value)

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

            # 4. Build structured tool log from all subtask records
            tool_log_lines: List[str] = []
            for st in subtasks:
                tool_log_lines.append(f"[子任務 {st.order}] {st.description}")
                tool_log_lines.append(f"  狀態: {st.status.value}")
                tool_log_lines.append(f"  結果: {(st.result or '無輸出')[:500]}")
                if st.error:
                    tool_log_lines.append(f"  錯誤: {st.error}")
            tool_log_text = "\n".join(tool_log_lines)

            had_escalation = any(
                getattr(st, 'status', None) == TaskStatus.ESCALATED for st in subtasks
            )
            results_texts = [st.result or "" for st in subtasks if st.result]

            # 5. Generate summary: Claude if escalated, Gemma otherwise
            if had_escalation or not results_texts:
                # 只有真的升級過才用 Claude 做 summary
                summary = await self.escalation.synthesize_summary(
                    req.goal, results_texts, task_id=req.id
                )
            else:
                # 全部地端完成 → 用 Gemma 讀取工具記錄，產出中文完成報告
                summary = await self._gemma_summarize(req.goal, tool_log_text, task_id=req.id)

            # 6. 任務完成後自動執行確認指令，把結果附加到 summary
            confirmation_output = await self._run_confirmation_checks(
                req.goal, subtasks=subtasks,
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
        base_prompt = _plan_prompt_for_mode(mode_value)

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
        raw_plan = await self._gemma_complete(
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
        """用 Gemma 讀取工具執行記錄，產出結構化的繁體中文完成報告。"""
        system_prompt = """\
你是一個任務執行報告生成器。根據提供的工具執行記錄，生成清楚的繁體中文完成報告。

規則：
1. 根據子任務的狀態和結果，清楚說明哪些步驟完成了、哪些失敗了
2. 不要說「我是 AI 我不能做...」，只描述實際發生的事情
3. 使用以下格式輸出：
   ✅ 完成的步驟（條列已成功完成的操作）
   ❌ 失敗的步驟（如果有，說明步驟和原因）
   📋 當前狀態（一段話總結整體結果）
4. 如果執行記錄中有 docker ps、ls、pip list 等驗證結果，把它納入報告
5. 保持簡潔，不要重複相同的資訊
6. 完全使用繁體中文回答
7. 【非常重要】只能描述「工具執行記錄」中實際出現過的容器、檔案、錯誤。
   不要提及記錄裡沒有的容器名稱 (例如任何未出現在本次 log 的舊容器)、
   也不要把主機上其他無關的狀態當成本次任務的失敗。
   如果某個錯誤訊息的容器名稱跟本次任務的目標不相關，請明確指出
   「此為主機既有狀態，與本次任務無關」。
"""
        prompt = (
            f"用戶的目標：{goal}\n\n"
            f"工具執行記錄：\n{tool_log}\n\n"
            "請根據以上記錄生成完成報告。"
        )
        result = await self._gemma_complete(
            prompt, system_prompt,
            task_id=task_id,
            task_description=f"[summary] {goal[:100]}",
        )
        # Fallback if Gemma returns empty
        if not result or result.strip() in ("[]", ""):
            lines = [f"任務已完成：{goal}", "", tool_log[:1000]]
            return "\n".join(lines)
        return result

    async def _run_confirmation_checks(
        self, goal: str, subtasks: Optional[List[SubTask]] = None,
    ) -> str:
        """根據任務目標關鍵字，執行確認指令並回傳格式化結果。

        對於 Docker 任務，會從子任務描述中擷取 compose 路徑（``cd X`` 或
        ``-f path``），優先使用 ``docker compose ps`` 只看本次任務的服務，
        不汙染主機上其他既有容器。擷取不到才退回全域 ``docker ps``。
        """
        goal_lower = goal.lower()
        confirmation_cmds: List[Tuple[str, str]] = []

        if "docker" in goal_lower:
            scoped_cmd = None
            for st in (subtasks or []):
                scoped_cmd = self._compose_scoped_ps_cmd(st.description or "")
                if scoped_cmd:
                    break
            if scoped_cmd:
                confirmation_cmds.append(("本任務容器狀態", scoped_cmd))
            else:
                confirmation_cmds.append(("Docker 容器狀態", "docker ps"))
        if any(kw in goal_lower for kw in ["clone", "git", "repository", "repo"]):
            confirmation_cmds.append(("目錄內容", "ls -la ~/ 2>/dev/null || ls -la ."))
        if any(kw in goal_lower for kw in ["install", "pip", "package"]):
            confirmation_cmds.append(("已安裝套件（前20個）", "pip list 2>/dev/null | head -20"))
        if any(kw in goal_lower for kw in ["npm", "node", "yarn"]):
            confirmation_cmds.append(("Node 套件", "ls node_modules 2>/dev/null | head -10 || echo 'node_modules not found'"))

        if not confirmation_cmds:
            return ""

        output_parts: List[str] = ["\n\n---\n📋 **確認狀態**"]
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

    async def _gemma_complete(
        self, prompt: str, system: Optional[str] = None,
        task_id: str = "", task_description: str = "[orchestrator]",
    ) -> str:
        payload = {
            "model":   config.gemma_model,
            "prompt":  prompt,
            "system":  system or _PLAN_SYSTEM_BASE,
            "stream":  False,
            "options": {
                "temperature": float(getattr(config, "gemma_temperature", 0.1)),
                "num_predict": config.gemma_max_tokens,
            },
        }
        try:
            resp = await self._http.post(
                f"{config.gemma_base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            # ── Record Ollama usage (best-effort) ─────────────────────────────
            tokens_in  = data.get("prompt_eval_count", 0) or 0
            tokens_out = data.get("eval_count", 0) or 0
            if tokens_in or tokens_out:
                try:
                    await self.tracker.record_usage(
                        task_id=task_id,
                        model=config.gemma_model,
                        model_provider="ollama",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        task_description=task_description,
                    )
                except Exception:  # noqa: BLE001
                    pass

            return data.get("response", "")
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
        logger.warning(
            "Could not parse Gemma plan JSON (raw length=%d); "
            "falling back to single subtask from goal.",
            len(raw),
        )
        if goal:
            return [
                SubTask(
                    parent_task_id=task_id,
                    description=f"執行用戶的原始請求: {goal}",
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

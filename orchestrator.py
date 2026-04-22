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

_PLAN_SYSTEM_BASE = """\
你是 Teddy-Orch，OpenTeddy 多智能體系統的任務規劃 AI。
把用戶的目標拆解成具體的、可執行的子任務。

每個子任務必須對應一個具體操作，例如：
- 執行 shell 指令：git clone / docker compose up / pip install 等
- 讀取檔案：執行 cat README.md 或 ls -la 查看結構
- 寫入檔案：寫入設定、建立腳本

【重要規則】：
- 每個子任務要具體到可以直接執行，不要模糊的描述
- 不要說「分析專案」，要說「執行 ls -la 和 cat README.md 查看專案結構」
- 不要說「設定環境」，要說「執行 cp .env.example .env 建立設定檔」
- 包含驗證步驟：做完每個重要操作後，加入一個子任務執行指令確認結果
- 子任務數量控制在 10 個以下

【重要輸出規則】：
- 最多拆 3-5 個子任務，不要超過 5 個
- 每個子任務描述不超過 50 字
- 輸出純 JSON 陣列，不要加任何說明文字、markdown、code block
- 格式範例：[{"description":"執行 git clone https://github.com/xxx","skill_hint":null,"order":0}]

只輸出 JSON 陣列，不要其他文字。每個元素包含：
  - "description": string  (具體操作描述，寫明要執行的指令或操作)
  - "skill_hint": string or null  (技能名稱，如果有對應技能則填入)
  - "order": integer  (執行順序，從 0 開始)
"""


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

        # Persist the task
        await self.tracker.create_task(req)
        await self.tracker.update_task_status(req.id, TaskStatus.RUNNING)

        try:
            # 1. Plan (with optional memory context)
            subtasks = await self._plan(req)
            for st in subtasks:
                await self.tracker.create_subtask(st)

            # 2. Execute subtasks sequentially (could be made parallel later)
            skills_used: list[str] = []
            new_skills:  list[str] = []

            for st in subtasks:
                st = await self._run_subtask(st, req.context)

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
            confirmation_output = await self._run_confirmation_checks(req.goal)
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
        memory_ctx = ""
        if self.memory is not None:
            try:
                memory_ctx = await self.memory.get_context_for_task(req.goal)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Memory retrieval failed (non-fatal): %s", exc)

        # Build a system prompt that includes memory when present
        system_prompt = _PLAN_SYSTEM_BASE
        if memory_ctx:
            system_prompt = memory_ctx + "\n\n" + _PLAN_SYSTEM_BASE

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

    async def _run_subtask(self, st: SubTask, context: Dict[str, Any]) -> SubTask:
        """Execute one subtask with retry; only escalate to Claude after all retries fail.

        Each attempt is wrapped with ``asyncio.wait_for(timeout=subtask_timeout)``
        so that a hung local model (Qwen tool-call freeze) is detected and
        immediately escalated to Claude instead of blocking forever.

        On success, appends a verification result if a matching check exists.
        """
        max_local_retries = getattr(config, 'escalation_failure_limit', 3)
        subtask_timeout   = getattr(config, 'subtask_timeout', 120)
        escalation_threshold = getattr(config, 'escalation_confidence_threshold', 0.6)

        for attempt in range(max_local_retries):
            try:
                st = await asyncio.wait_for(
                    self.executor.execute(st, context),
                    timeout=float(subtask_timeout),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Subtask %s timed out after %ds on attempt %d/%d — escalating to Claude.",
                    st.id, subtask_timeout, attempt + 1, max_local_retries,
                )
                st.status = TaskStatus.FAILED
                st.error  = (
                    f"地端模型執行超時（>{subtask_timeout}s），自動升級到 Claude"
                )
                await self.tracker.update_subtask(st)
                # 超時直接升級，不再重試
                st = await self.escalation.resolve(st, context)
                return st

            confidence = getattr(st, 'confidence', 1.0)

            if st.status != TaskStatus.FAILED and confidence >= escalation_threshold:
                # 地端成功 → 嘗試執行驗證指令
                verification = await self._verify_subtask(st)
                if verification:
                    st.result = (st.result or "") + f"\n\n[驗證結果]\n{verification}"
                    await self.tracker.update_subtask(st)
                return st

            logger.info(
                "Subtask %s local attempt %d/%d failed (conf=%.2f), retrying...",
                st.id, attempt + 1, max_local_retries, confidence,
            )

        # 所有重試都失敗 → 才升級到 Claude
        logger.info(
            "Subtask %s escalating to Claude after %d failed attempts.",
            st.id, max_local_retries,
        )
        st = await self.escalation.resolve(st, context)
        return st

    async def _verify_subtask(self, st: SubTask) -> Optional[str]:
        """根據子任務類型，執行對應的驗證指令，回傳驗證輸出（或 None）。"""
        desc = st.description.lower()

        # 對應關鍵字 → 驗證指令
        checks: List[Tuple[List[str], str]] = [
            (["git clone"],                    "ls -la"),
            (["docker compose", "docker-compose", "docker run"],  "docker ps"),
            (["docker"],                       "docker ps"),
            (["pip install"],                  "pip list | head -20"),
            (["npm install", "yarn install"],  "ls node_modules 2>/dev/null | head -10 || echo 'node_modules not found'"),
            (["mkdir"],                        "ls -la"),
            (["cp ", "copy "],                 "ls -la"),
        ]

        for keywords, cmd in checks:
            if any(kw in desc for kw in keywords):
                logger.info("Running verification '%s' for subtask %s", cmd, st.id)
                try:
                    proc = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout_b, stderr_b = await asyncio.wait_for(
                        proc.communicate(), timeout=10.0
                    )
                    out = stdout_b.decode(errors="replace").strip()
                    err = stderr_b.decode(errors="replace").strip()
                    return (out or err or "(no output)")[:800]
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Verification command failed (non-fatal): %s", exc)
                    return None

        return None  # 沒有匹配的驗證規則

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

    async def _run_confirmation_checks(self, goal: str) -> str:
        """根據任務目標關鍵字，執行確認指令並回傳格式化結果。"""
        goal_lower = goal.lower()
        confirmation_cmds: List[Tuple[str, str]] = []

        if "docker" in goal_lower:
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
            "options": {"temperature": 0.1, "num_predict": config.gemma_max_tokens},
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

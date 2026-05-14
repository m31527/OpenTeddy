<div align="center">

<sub>English | <a href="README.zh-TW.md">繁體中文</a></sub>

<img src="static/OpenTeddy-logo.svg" alt="OpenTeddy" width="240" />

# OpenTeddy

**The platform that makes local LLMs ship work.**

Local models alone are weak. Wrap them in OpenTeddy and you get a real agent —
hardened orchestration, a self-growing skills library, and just enough
commercial-LLM escalation to finish what local can't.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" />
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-local-black?logo=ollama&logoColor=white" />
  <img alt="Anthropic" src="https://img.shields.io/badge/Claude-Anthropic-D97757" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green" />
  <a href="https://github.com/m31527/OpenTeddy/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/m31527/OpenTeddy?style=social" /></a>
</p>

🌐 **Web:** [https://openteddy.net/](https://openteddy.net/) &nbsp;·&nbsp; 📦 **Source:** [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy)

### 📥 Download

| | |
|:-:|:-:|
| 🍎 **macOS desktop** | [`OpenTeddy-1.0.1-aarch64.dmg`](https://openteddy.net/download/mac) (105 MB, Apple Silicon, signed + notarized) |
| 🐧 **Linux / WSL2** | `curl -fsSL https://openteddy.net/install \| bash` |
| 🐳 **Docker** | see [Docker Deployment](#docker-deployment) below |

[View all releases ↗](https://github.com/m31527/OpenTeddy/releases)

</div>

---

## Why this exists

A 2B / 4B / 7B local model on its own is a toy. It hallucinates, it loops,
it stops mid-task. The model isn't the product — **the platform around it
is**. OpenTeddy is that platform:

- A **hardened agent loop** that knows when to give up, when to retry, and
  when to call Claude — no infinite "let me try that again" doom-spirals.
- A **self-growing skills library** that turns repeated work into plain
  Python functions, so the second time you ask the same question it costs
  zero LLM calls.
- **Hardware-tuned model presets** for everything from a 16 GB MacBook to
  a DGX Spark — the right `num_ctx`, `max_tokens`, and timeout per tier.
- **Commercial-LLM escalation as a safety net**, not a bill — Claude only
  gets called when local genuinely can't finish, and the Usage dashboard
  shows you how much GPT-4 would've charged for the same work.

The result: your $0/token local hardware actually finishes the job, and
the savings counter in the sidebar is what makes you stop worrying about
Claude Pro auto-renewing.

> **If this resonates with you — or you just want to cheer the project on —
> please drop a ⭐ on the repo. It genuinely helps and keeps me motivated to
> ship more.** → [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy)

## Highlights

- **Three LLM modes, one toggle** — *Local only* / *Mixed* (default — local with cloud safety net) / *Full Cloud LLM* (skip Ollama entirely, route every subtask through your cloud provider). Pick per-app in Settings; per-session "Local only" override always wins for privacy-sensitive work.
- **Five cloud LLM providers** — Anthropic Claude / OpenAI / Google Gemini / Deepseek / OpenRouter, swappable from Settings at runtime. Usage tab attributes spend per provider.
- **Auto-escalation safety net** — timeouts, low confidence, repeated failures, hard-failure signals in tool output (`unhealthy` containers, `ERROR 1045`, `command not found`), or "the task asked for a file but the model produced zero" all trigger cloud-LLM intervention automatically.
- **PDF analysis** — drop a `.pdf` into chat, ask questions; the agent extracts text page-by-page (CJK supported) and can cite page numbers. Image-only PDFs are flagged honestly rather than fabricated.
- **Connect a database to any session** — Postgres / MySQL / SQLite / MSSQL / Oracle / DuckDB via SQLAlchemy. Two-step Test → Connect modal. Destructive SQL (`DELETE` / `DROP` / `TRUNCATE` / `UPDATE`) is hard-blocked on every code path — defence in depth.
- **Per-session workspace isolation** — each new session gets its own `agent-workspace/sessions/<id>/`; files from one session never bleed into another. Toggleable in Settings.
- **Self-growing skills** — repeated tasks are promoted into reusable Python skills, cutting LLM calls over time.
- **Streaming UI** — both the orchestrator's planning and the executor's answer stream token-by-token via WebSocket — no more staring at a spinner while the model thinks.
- **Per-step deliverable verification** — LLM-as-judge confirms each produced file actually matches the goal, catching the "wrote a report *about* the game instead of the game" failure mode. Toggleable for big-model setups where extra calls are too costly.
- **Loop hardening for small models** — adaptive prompts, a parallel low-risk tool fan-out, per-tool-name caps, a circuit breaker, discovery memos, a context watchdog that compresses old turns before busting `num_ctx`, and pinned session-workspace context that prevents "model wanders to the wrong directory" drift.
- **Forever-command guard** — shell tool refuses `tail -f`, `journalctl -f`, `watch …`, and auto-adds `-d` to `docker compose up` so a runaway log stream can't hang an entire subtask.
- **Web search built in** — Chat mode exposes the `web_search` tool (Brave Search API) so the local model can ground answers in current data instead of hallucinating recent events / version numbers / today's prices.
- **Reconnect-safe streaming** — the WebSocket carries a 600-event ring buffer so a flaky network or a tab refresh replays the missed events instead of leaving the UI stuck.
- **Web dashboard** — submit tasks, watch tool calls stream live, review pending approvals, manage memory, render Markdown/GFM tables, embed Chart.js datalabels in HTML reports, see live "Saved $X vs GPT-4" in the sidebar, and tune settings.
- **Persistent artifact chips** — files produced by a task keep their 📎 download + 👁 preview buttons after switching sessions, closing the tab, or opening from another device.
- **Performance observability** — Usage tab shows per-mode (Chat / Code / Analytic) time-to-result stats (mean / p50 / p95 / max) plus your slowest 10 tasks. Screenshot it into a bug report when something looks off.
- **Capabilities tab** — built-in Tools and learned Skills merged into one filterable list with type badges, so users see "what can my agent do?" in one place. Skills graduate from `TESTING` to `ACTIVE` automatically; the count grows as the install matures.
- **Auto-approve countdown** — opt-in setting that turns the high-risk approval gate fail-permissive after N seconds, with a visible amber timer so you have time to Deny if the agent's about to do something unexpected. Default: off.
- **Native macOS desktop client** — Tauri 2.x shell with onboarding wizard (Ollama install + tier-based model pull), language picker, mode-locked sessions, full window-drag regions, auto-update against GitHub Releases, and one-click diagnostics download. See [`desktop/`](desktop/).
- **Optional cloud account** — sign in with Google to enable cross-device sync (skills, memory, settings) and licence-gated premium content. Anonymous Firebase auth runs from day one; the upgrade is opt-in. The OSS web UI stays Firebase-free — auth lives only in the desktop shell.
- **Lifetime licensing via Lemon Squeezy** — one-time $99 unlocks the polished signed desktop, cloud sync, and (future) premium skill packs. Open-source core stays free forever for everyone. Webhook-driven licence activation: the sidebar pill flips from "Get Lifetime" to your account email within ~1 sec of payment clearing.
- **Analytic / report mode** — first-class `csv_describe` + `python_exec` tools and an HTML report generator that renders charts with value labels.
- **Human-in-the-loop** — high-risk shell commands (rm, sudo, mv, …) pause for approval before running.
- **Persistent memory** — ChromaDB-backed long-term memory feeds relevant context back into future plans.
- **22-locale i18n** — UI strings live in `static/i18n.js`; build-hash check + per-commit cache-buster auto-reload when the dashboard is updated.
- **Hot-reloadable settings** — change models, thresholds, performance toggles, API keys (Anthropic, Brave Search, Lemon Squeezy), and endpoints from the UI without restarting the server.

## Architecture

```
User Goal
   │
   ▼
┌───────────────────────────────────────────────────┐
│  Orchestrator  (Gemma via Ollama)                  │
│  • Decomposes goal into ordered SubTasks           │
│  • Streams plan tokens to the UI as it thinks      │
│  • Retrieves long-term memory for context          │
│  • Drives execution + escalation loop              │
└────────────────────┬──────────────────────────────┘
                     │ SubTasks
                     ▼
┌───────────────────────────────────────────────────┐
│  Executor  (Qwen via Ollama, function calling)     │
│  • Runs a matching Skill if available              │
│  • Uses tools: shell, file (incl. pdf_extract_text),│
│    http, db, gcp, package, csv_describe,           │
│    python_exec, generate_report, web_search        │
│  • Streams answer tokens; parallelises low-risk    │
│    tool calls; caps per-tool-name retries          │
│  • Compresses old turns when context fills up      │
│  • Reports confidence (clamped on hard failures)   │
└────────────────────┬──────────────────────────────┘
                     │ produced files
                     ▼
┌───────────────────────────────────────────────────┐
│  Deliverable Verifier  (LLM-as-judge, Qwen)        │
│  • Reads the produced HTML/MD/Py/etc.              │
│  • Verdict: PASS or FAIL — forces retry on FAIL    │
│  • Skipped via `verification_enabled = false`      │
└────────────────────┬──────────────────────────────┘
    low conf │ timeout │ failure signal │ unhealthy
                     ▼
┌───────────────────────────────────────────────────┐
│  Escalation Agent  (Claude via API)                │
│  • Resolves hard subtasks with full diagnostics    │
│  • Synthesises the final summary                   │
└────────────────────┬──────────────────────────────┘
                     ▼
┌───────────────────────────────────────────────────┐
│  Skill Factory  (Claude via API)                   │
│  • Generates new Python skills on demand           │
│  • Promotes skills after N successes               │
│  • Saves skills to disk + SQLite DB                │
└───────────────────────────────────────────────────┘
```

## Loop Hardening (small-model resilience)

The agent loop has been progressively hardened to make small / mid-size local
models (Gemma 3:4B, Qwen 2.5:3B class) reliable enough to ship work end-to-end,
not just fast enough to look impressive on a single tool call:

| Mechanism | What it does |
|---|---|
| **Adaptive prompts** | Compact system prompts on small models; richer guidance only when context allows. |
| **Parallel tool fan-out** | Low-risk tool calls (file reads, shell `ls`, HTTP gets, `csv_describe`) inside a single round are dispatched with `asyncio.gather` instead of serially. |
| **Per-step deliverable verification** | After each successful subtask, an LLM-as-judge reviews the produced HTML/MD/code file. If it looks like a *description of* the goal rather than the actual deliverable (the "Snake Game report" failure pattern), the subtask is forced to retry with feedback. |
| **Context watchdog** | When the prompt size approaches `num_ctx`, the executor compresses earlier turns into a recap and pins discovery memos to the system prompt — keeping recent tool context intact instead of letting Ollama silently truncate. |
| **Discovery memos** | Useful one-off facts learned from tool calls (e.g. "the workspace already contains `data.csv` with columns X/Y/Z") are pinned to the system prompt so the model doesn't re-discover them every round. |
| **Per-tool-name cap (tiered)** | Read-only inspection tools (`read_file`, `list_directory`, `db_query`, `db_describe_table`, `csv_describe`, `pdf_extract_text`, `web_search`, `shell_exec_readonly`) get a cap of 10–15. State-mutating tools (`write_file`, `python_exec`, `shell_exec_write`, …) stay at 5 — they're the ones that actually loop. |
| **Empty-artifact guard** | When a code/analytic subtask description mentions building / creating / writing a concrete file but workspace got 0 new artifacts, confidence is clamped and the loop forces a retry (eventually escalates). Catches "small model wrote a confident summary without actually producing the deliverable" — the deliverable judge can't catch this case because it has no file to look at. |
| **macOS PATH augmentation** | Shell subprocesses get `/opt/homebrew/bin`, `/usr/local/bin`, `~/.cargo/bin`, `/Applications/Docker.app/Contents/Resources/bin`, … appended to PATH. Tauri's minimal LaunchServices PATH would otherwise leave the agent with `docker: command not found` on a Mac that obviously has Docker installed. |
| **Circuit breaker** | After 5 cumulative tool failures the loop is forced to commit to a final answer instead of looping forever. |
| **Common error hints** | Twelve frequent stack-trace patterns (`ModuleNotFoundError`, `KeyError`, `PermissionError`, …) are matched against tool stderr and converted into one-line hints so the model corrects itself instead of repeating the same mistake. |
| **WS reconnect + replay** | The dashboard WebSocket carries a 600-event ring buffer keyed by sequence number — a refreshed tab or a wifi blip replays missed events on reconnect. |
| **Pinned workspace context** | Every executor round prepends `WORKSPACE: <abs-path>` to the user message, and shell-tool refusals embed the correct path — small models stop drifting to "their idea of the project root" and re-emitting `working_dir=/home/.../OpenTeddy` round after round. |
| **Forever-command guard** | `_sanitize_command` auto-adds `-d` to `docker compose up`, strips `-f`/`--follow` from `docker logs` / `docker compose logs`, and refuses `tail -f`, `journalctl -f`, `watch …` outright. Stops a container in a restart-crash loop from holding a subtask hostage forever. |
| **Web-search grounding** | Chat mode exposes the `web_search` tool (Brave Search) so the local model can ground answers in current data instead of hallucinating events / version numbers / prices past its training cutoff. |

## File Structure

```
OpenTeddy/
├── config.py          # Config via .env / environment variables
├── models.py          # Pydantic models + SQLite schema
├── tracker.py         # Async SQLite persistence (aiosqlite) + perf stats
├── skill_factory.py   # Claude-powered skill generation & loader
├── executor.py        # Qwen executor — function calling, streaming,
│                      #   parallel low-risk tools, context watchdog,
│                      #   discovery memos, per-tool cap, circuit breaker
├── escalation.py      # Claude escalation agent
├── orchestrator.py    # Gemma orchestrator (plan → execute → verify →
│                      #   escalate) + per-step deliverable judge
├── memory.py          # ChromaDB long-term memory
├── approval_store.py  # Human-in-the-loop approval queue
├── settings_store.py  # Hot-reloadable settings (SQLite-backed)
├── tool_registry.py   # Tool registration + risk gating
├── tools/             # shell / file / http / db / gcp / package /
│                      #   analytic (csv_describe, python_exec) /
│                      #   report_tool (HTML + Chart.js datalabels)
├── skills/            # Auto-generated skill .py files
├── static/            # Web dashboard (index.html, i18n.js — 22 locales,
│                      #   OpenTeddy-logo.svg)
├── desktop/           # Native macOS Tauri 2.x client (own repo)
├── main.py            # FastAPI server + CLI entry point + WS ring buffer
└── .env.example       # Environment variable template
```

## Install in one line

```bash
curl -fsSL https://openteddy.net/install | bash
```

The installer:
- checks Python 3.10+, git, curl
- detects Ollama (warn-only if missing — cloud LLMs still work)
- `git clone`s OpenTeddy to `~/OpenTeddy`
- creates `.venv` + `pip install -r requirements.txt`
- pulls the two default Ollama models (`gemma4:e2b`, `qwen3.5:2b`) if Ollama is present
- prints next-steps instructions

It's **idempotent** — re-run any time to pull the latest source + refresh deps. All work scoped to `$HOME/OpenTeddy`, no `sudo`, no secondary scripts fetched.

Want to audit before running? `curl -fsSL <url> -o install.sh && less install.sh && bash install.sh` is encouraged. The script also accepts `--dry-run` to preview what it would do without changing anything.

After install:

```bash
cd ~/OpenTeddy
./run.sh --open      # boots uvicorn on :8000 + opens browser
```

Customisation flags: `--dir <path>`, `--force`, `--skip-models`. See `./install.sh --help`.

## Quick Start (manual)

If you'd rather install by hand:

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally (optional — skip this if you'll use cloud LLMs only):
  ```bash
  ollama pull gemma4:e2b
  ollama pull qwen3.5:2b
  ```
- (Optional) An Anthropic / OpenAI / Gemini / Deepseek / OpenRouter API key — configure later via Settings → Cloud LLM Provider

### 2. Install

```bash
git clone https://github.com/m31527/OpenTeddy.git
cd OpenTeddy
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY
```

### 4. Run

```bash
uvicorn main:app --reload
# Dashboard → http://localhost:8000
# API docs  → http://localhost:8000/docs
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/run` | Submit a task |
| `GET`  | `/tasks/{id}` | Check task status |
| `GET`  | `/tasks` | List recent tasks (filter by `session_id`) |
| `GET`  | `/skills` | List all skills |
| `POST` | `/skills/generate?name=…&description=…` | Manually create a skill |
| `GET`  | `/tools` | List available tools |
| `GET`  | `/approvals` | Pending human approvals |
| `POST` | `/approvals/{id}/approve` \| `/reject` | Resolve an approval |
| `GET`  | `/memory` | Browse long-term memory |
| `GET`  | `/usage`, `/usage/summary` | Token usage & estimated cost |
| `GET`  | `/benchmark/stats` | Per-model token-throughput stats (#6) |
| `GET`  | `/settings` \| `POST` `/settings` | Read/update runtime settings |
| `GET`  | `/settings/ollama/models` \| `/status` | Local model management |
| `POST` | `/settings/ollama/pull` | Pull a model (streamed progress) |
| `GET`  | `/version` | Build hash + version (used by UI auto-reload) |
| `GET`  | `/update/check` | Check GitHub Releases for a newer version |
| `POST` | `/update/apply` | Apply an available update |
| `POST` | `/optimize_prompt` | Rewrite a draft goal via Claude |
| `GET`  | `/admin/diagnostics` | Download a zipped diagnostic bundle |
| `GET`  | `/health` | Health check |
| `WS`   | `/ws?since=N` | Live event stream — `since` replays the ring buffer from sequence `N` |

### Example request

```bash
curl -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Summarise the key benefits of async Python", "priority": 7}'
```

## How Claude Steps In

OpenTeddy tries to keep every task local. Claude is called **only** when the local path breaks down:

| Trigger | Where | Default |
|---------|-------|---------|
| Subtask timeout (local model hangs) | `orchestrator._run_subtask` | 120 s |
| Low self-reported confidence | `executor._qwen_execute` | `< 0.6` |
| Repeated failures in a row | `orchestrator._run_subtask` | 3 |
| Hard-failure signal in tool output (`unhealthy`, `Exited`, `ERROR 1045`, `Error response from daemon`, …) | `executor._finalize_response` | confidence clamped to 0.3 → escalates |
| Container health check fails after a Docker task | `orchestrator._inspect_docker_health` | auto-pulls `docker logs` + `inspect`, then escalates |
| Deliverable verifier returns `FAIL` | `orchestrator._verify_deliverable` | confidence clamped to 0.3 → retry, then escalate |
| Circuit breaker tripped (5 cumulative tool failures) | `executor._qwen_execute` | forces final-answer commit; escalation kicks in if confidence is still low |

This keeps cost low for everyday work while still guaranteeing you get a real answer when the local model cannot deliver one. Two ways to opt out:

- **Settings → LLM Mode → Local only** — global, applies to every session
- **Per-session "Local only" toggle** in the session row — overrides the global mode for that specific session, so you can run a private analysis inside a Cloud-mode app without the data ever leaving your machine

Set `OPENTEDDY_LLM_MODE=local` in `.env` if you want the local-only choice baked in across restarts.

## Self-Growth Mechanism

1. When Qwen executes a subtask it suggests a **skill name** if a reusable function would have helped.
2. The Executor calls `SkillFactory.generate_skill()` in the background.
3. Claude writes the skill as an `async def run(input_data: dict) -> str` function and saves it to `skills/<name>.py`.
4. The skill starts in **TESTING** status. After `SKILL_PROMOTION_THRESHOLD` successful invocations it is promoted to **ACTIVE**.
5. Future tasks automatically match and invoke active skills — no LLM call needed.

## Pricing & licensing

OpenTeddy ships under an **open-core** model — the OSS backend stays MIT-
licensed and free forever. The paid tier is the polished native desktop
experience plus a few cloud-backed conveniences:

| | **Free** (this repo) | **Lifetime — $99 once** |
|---|---|---|
| Backend (FastAPI + orchestrator + executor + tools) | ✅ | ✅ |
| Local models via Ollama | ✅ | ✅ |
| Loop hardening (verifier, watchdog, circuit breaker, parallel fan-out) | ✅ | ✅ |
| Self-growing skills | ✅ | ✅ |
| 22-locale i18n web dashboard | ✅ | ✅ |
| Self-build the desktop from source | ✅ | ✅ |
| Signed `.dmg` + auto-updates against GitHub Releases | ❌ | ✅ |
| Cross-device cloud sync (skills, memory, settings) | ❌ | ✅ |
| Premium skill packs (planned: Analytics / Marketing / Memory Pro) | ❌ | ✅ |
| Priority bug-fix queue + private support channel | ❌ | ✅ |
| Includes Claude API credits to get started | ❌ | ✅ (planned) |

The whole identity + billing stack is built in stages, all visible in this
codebase:

- **Phase A** — Anonymous Firebase Auth on app start; `users/{uid}` doc tracks
  device id, platform, app version, last-seen.
- **Phase B** — Real Google sign-in via system-browser pairing (Tauri WebKit
  can't do `signInWithPopup`, so we open the system browser, run the popup
  there, and pass a custom token back via Firestore + a Cloud Function).
  Account-merge carries the anonymous user's `deviceId` + `createdAt` onto
  the new Google identity.
- **Phase C** — Lemon Squeezy checkout flow. The desktop builds a buy URL with
  `checkout[email]` and `checkout[custom][uid]` prefilled; the
  `lemonSqueezyWebhook` Cloud Function verifies HMAC, looks up the buyer
  (by `custom_data.uid` or email fallback), and writes
  `users/{uid}.subscription = { status: 'active', plan: 'lifetime' }` plus
  a canonical `licenses/{uid}` record. The desktop's Firestore listener
  flips the upgrade pill off within ~1 sec of payment clearing.

All Cloud Function code, Firestore Security Rules, and the auth bridge HTML
live in [`desktop/`](desktop/) (functions/) and [`landing-page/`](https://openteddy.net/auth)
respectively. Stripe-style webhook signatures are verified against
HMAC-SHA256 of the raw body using a per-endpoint secret stored in Google
Secret Manager (`firebase functions:secrets:set LEMONSQUEEZY_WEBHOOK_SECRET`).

Anyone running the OSS backend in a plain browser sees **none of this** —
the cloud-sync pill, upgrade pill, and sign-in dialog are gated on
`window.parent !== window` (i.e. running inside the Tauri shell), so the
Firebase JS bundle never loads on a self-hosted install.

## Configuration Reference

Most of these can also be edited live from the dashboard's **Settings** panel —
changes are persisted to SQLite and `config.reload_from_store()` re-applies them
without a server restart.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required only if escalation is enabled. Anthropic API key. |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Claude model for escalation. |
| `GEMMA_BASE_URL` | `http://localhost:11434` | Ollama base URL for the orchestrator. |
| `GEMMA_MODEL` | `gemma4:e2b` | Orchestrator model tag. |
| `QWEN_BASE_URL` | `http://localhost:11434` | Ollama base URL for the executor. |
| `QWEN_MODEL` | `qwen3.5:2b` | Executor model tag. |
| `BRAVE_SEARCH_API_KEY` | — | Optional. Powers the Chat-mode `web_search` tool. Free tier covers 2,000 queries/month at [api-dashboard.search.brave.com](https://api-dashboard.search.brave.com/). Without it, the local model answers from training data and warns the user about staleness. |
| `DB_PATH` | `openteddy.db` | SQLite database path. |
| `MEMORY_DB_PATH` | `./memory_db` | ChromaDB directory. |
| `SKILLS_DIR` | `skills` | Directory for skill files. |

### LLM Mode (canonical)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENTEDDY_LLM_MODE` | `mixed` | One of `local` / `mixed` / `cloud`. `local` = Gemma plans, Qwen executes, never call cloud. `mixed` = local-first with cloud safety net on failure. `cloud` = every subtask handled directly by the configured cloud LLM (Ollama not required). The legacy `ESCALATION_ENABLED` below is auto-derived from this. |
| `LLM_PROVIDER` | `anthropic` | Which cloud LLM provider escalation + Cloud mode route to. One of `anthropic` / `openrouter` / `openai` / `gemini` / `deepseek`. |

### Escalation

| Variable | Default | Description |
|----------|---------|-------------|
| `ESCALATION_ENABLED` | `true` | Legacy kill-switch — now auto-derived from `OPENTEDDY_LLM_MODE` (`local` → False, `mixed` / `cloud` → True). Setting this directly still works for backward compat. |
| `ESCALATION_THRESHOLD` | `0.6` | Min Qwen confidence before escalation. |
| `ESCALATION_FAILURE_LIMIT` | `3` | Max consecutive failures before escalation. |
| `SUBTASK_TIMEOUT` | `900` | Wall-clock seconds before a subtask is treated as hung. Real hang detection is via `SHELL_SILENCE_TIMEOUT`; this is just the ceiling. |
| `SHELL_SILENCE_TIMEOUT` | `180` | Kill a shell command after this many seconds of no stdout/stderr output. Long-but-active commands (docker build, pip install) stay alive as long as they're printing progress. |
| `SKILL_PROMOTION_THRESHOLD` | `5` | Successes needed to promote a skill. |
| `APPROVAL_AUTO_APPROVE_AFTER` | `0` | Seconds after which a high-risk approval auto-resolves to **approved**. 0 = off (the safer default — wait for explicit click). |

### Performance toggles (loop hardening)

Most of these matter most on big models — turn them off to trade safety nets for speed.

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMING_ENABLED` | `true` | Stream LLM tokens to the chat as they generate. Major perceived-latency win on small thinking models. |
| `VERIFICATION_ENABLED` | `true` | Run the per-step LLM-as-judge verifier after each successful subtask. Set to `false` on big-model setups (DGX Spark, qwen3.5:35b) where each judge call is 5–60s. |
| `QWEN_NUM_CTX` | `16384` | Ollama `num_ctx` for the executor. Larger = more tool-round history before the watchdog has to compress, but more VRAM. |
| `GEMMA_NUM_CTX` | `16384` | Same, for the orchestrator. |
| `CONTEXT_COMPRESS_AT` | `0.7` | Trigger context compression when prompt-token usage crosses this fraction of `num_ctx`. |

## Desktop Client (macOS)

OpenTeddy ships with a native macOS shell built on **Tauri 2.x** that wraps the
web dashboard inside a polished launcher. Source lives in [`desktop/`](desktop/)
(its own repo — gitignored from the main repo).

What you get on top of the web UI:

- **Onboarding wizard** — language picker, Privacy Policy gate, hardware
  tier-select (Beginner / Advanced / Flagship), one-click Ollama install,
  streaming model-pull progress.
- **Mode-locked sessions** — once a session has its first task, the Chat /
  Analytic / Build mode is locked so the agent's tool palette stays consistent
  for that conversation.
- **Custom dialogs** — replaces native `confirm` / `alert` / `prompt` (which
  Tauri blocks) with in-app modals that match the chrome.
- **Anonymous-by-default Firebase Auth** — every install gets a stable Firebase
  uid + Firestore `users/{uid}` doc on first launch. Google sign-in is
  optional and unlocks cross-device cloud sync.
- **Sign-in dialog with system-browser pairing** — Google OAuth runs in real
  Safari/Chrome (Tauri's WebKit user-agent is blocked by Google's policy),
  the resulting Firebase customToken comes back to the desktop via a one-shot
  `pairings/{pairId}` Firestore doc with HMAC-style nonce verification.
- **Sidebar pills** — live "Saved $X vs GPT-4" running total, ☁️ cloud-sync /
  account email, and ✨ Get Lifetime $99 upgrade CTA (hidden once paid).
- **Lemon Squeezy checkout** — click ✨ Get Lifetime → opens checkout in the
  system browser with email + Firebase uid prefilled → webhook flips
  `subscription.status` to `active` → upgrade pill auto-disappears.
- **Auto-update against GitHub Releases** — periodic poll, in-app changelog,
  one-click apply.
- **Diagnostics download** — single-click `app.log` + tasks/usage/settings
  zip for bug reports.
- **Returning launches skip the splash** — once you've finished onboarding the
  splash goes straight into `enter_main`, so subsequent starts land on the
  main window immediately.

```bash
cd desktop
npm install
npx tauri dev          # hot-reload dev (still needs uvicorn running separately)

# Iteration: dev build (filename gets a "dev-" prefix so you can't
# accidentally mistake it for a public release)
./scripts/build_macos.sh

# Ship: real release — builds + signs + notarizes + git-tags + uploads
# to GitHub Releases in one shot. Requires APPLE_DEV_ID +
# APPLE_NOTARY_PROFILE in desktop/.notarize.env.
./scripts/release.sh 1.0.2
./scripts/release.sh 1.0.2 --dry-run   # walk through without acting
```

The shipping `.dmg` (from `release.sh`) is **signed with our Apple
Developer ID and notarized by Apple**, so the published builds work
with one double-click on any Mac — no Gatekeeper warnings, no
`xattr` workaround needed. (Self-built `.dmg`s from
`build_macos.sh` without the notarize env set are ad-hoc signed and
will hit Gatekeeper on machines other than the one that built
them — that's expected for dev iteration.)

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **macOS** (Intel / Apple Silicon) | ✅ Fully supported | Primary development target. |
| **Linux** | ✅ Fully supported | Any distro with Python 3.11+ and Ollama. |
| **Windows (native)** | ⚠️ Partial — use WSL2 if possible | See caveats below. |
| **Windows (WSL2)** | ✅ Fully supported | Behaves like Linux. Recommended on Windows. |

### Windows caveats

The codebase itself is cross-platform Python (uses `pathlib`, `os.path.join`,
`asyncio`), and `package_tool.py` already handles the Windows venv layout
(`Scripts\pip.exe`). The things that actually trip Windows users are:

- **The executor LLM generates POSIX shell commands.** When Qwen decides to
  run `ls`, `rm -rf`, `grep`, `chmod`, or pipes like `cmd1 | tee file`, those
  are executed through the system shell — which is **cmd.exe / PowerShell**
  on native Windows, so they fail. Running OpenTeddy under **WSL2** makes
  this a non-issue.
- **`lsof` / `ps` are not available** on native Windows. The deploy-tool
  helpers that inspect port occupancy (`port_probe`, `port_free` in
  [`tools/deploy_tool.py`](tools/deploy_tool.py)) degrade: `port_probe`
  returns a bound/free flag but no PID/process name; `port_free` returns an
  error and cannot kill by port.
- **Ollama on Windows** is officially supported (install from ollama.com) —
  pulling and running Gemma/Qwen works the same as on Mac/Linux.

**Recommendation:** on Windows, install Ollama natively on the host, then
run OpenTeddy itself inside **WSL2 Ubuntu**. That gives you GPU-accelerated
local inference + a POSIX userspace for the shell-heavy parts of the agent.

### Docker network caveat (Linux hosts)

`docker-compose.yml` uses `extra_hosts: ["host-gateway:host-gateway"]` so
the container can reach Ollama running on the host. This requires Docker
Engine **20.10+** on Linux, and Ollama must be bound to `0.0.0.0`, not
just `127.0.0.1` — otherwise the container's bridged traffic can't reach
it. Set `OLLAMA_HOST=0.0.0.0:11434` before `ollama serve`. On Docker
Desktop (Mac / Windows) this "just works".

## Docker Deployment

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY
docker compose up -d
# Open http://localhost:8000
```

Notes:

- Ollama must be running on the host (`ollama serve`).
- The container reaches host Ollama via the `host-gateway` alias set in `docker-compose.yml`.
- Skills and the usage database persist in the `openteddy_data` Docker volume.
- Rebuild image: `docker compose up -d --build`.

### ⚠️ Docker cannot touch your host filesystem

The default `docker-compose.yml` only mounts an isolated named volume
(`openteddy_data` → `/app/data`). It does **not** bind-mount your home
directory, Desktop, Downloads, or any other host folder. That means:

- Tasks like *"read `~/Documents/report.pdf`"*, *"tidy up my Downloads folder"*,
  or *"run this script on my Desktop"* **will not work** in the Docker setup —
  the container simply cannot see those files.
- The agent's shell/file/python tools operate entirely **inside** the
  container. Any files it reads or writes live in `/app/data` and disappear
  if the volume is removed.

**If you need the agent to operate on files on your machine, run OpenTeddy
directly with `uvicorn` (see [Quick Start](#quick-start)) instead of Docker.**
The native process has full access to your filesystem (subject to your user's
permissions), which is what most "local assistant" use cases actually want.

Alternatively, if you really want to stay on Docker, you can add a bind mount
to `docker-compose.yml` — e.g.:

```yaml
    volumes:
      - openteddy_data:/app/data
      - ${HOME}/openteddy-workspace:/workspace   # ← exposed host folder
```

…and then point the agent at `/workspace` inside the container. Only the
folders you explicitly mount are visible; everything else stays isolated.

## Support the project

OpenTeddy is a solo side-project trying to prove that a small open stack can
get close to the big commercial agents. If you want to see it keep growing:

- ⭐ **Star the repo** — [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy) —
  it's the single biggest encouragement I get.
- 🐛 **Open an issue** if something breaks or a model setup confuses you.
- 🧠 **Share a skill** you built on top of OpenTeddy — PRs welcome.

## License

MIT

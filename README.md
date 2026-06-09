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
| 🍎 **macOS desktop** | [`OpenTeddy-1.0.2-aarch64.dmg`](https://openteddy.net/download/mac) (105 MB, Apple Silicon, signed + notarized) |
| 🐧 **Linux desktop**  *(NEW)* | `.AppImage` / `.deb` for x86_64 — see [Releases](https://github.com/m31527/OpenTeddy/releases) (auto-built via GitHub Actions on every tag) |
| 🐧 **Linux / WSL2 OSS** | `curl -fsSL https://openteddy.net/install \| bash` |
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
- **8+ more document formats via `doc_to_markdown`** *(v1.1.0)* — PowerPoint, Word, Excel, EPUB, images (EXIF + OCR), audio (EXIF + transcription), HTML, CSV/JSON/XML, ZIP archives, and YouTube URLs — all read through a single tool backed by [Microsoft markitdown](https://github.com/microsoft/markitdown). PDFs unchanged (pypdf still canonical, A/B tested better on resumes / forms).
- **755 expert workflows via `cyber_skill_lookup`** *(v1.1.0)* — indexed catalogue of cybersecurity workflows from [Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills) (754 entries, mapped to MITRE ATT&CK / NIST CSF / D3FEND / ATLAS / NIST AI RMF) plus [last30days-skill](https://github.com/mvanhorn/last30days-skill) for multi-platform trend research. Agent auto-consults the catalogue first when goals touch security / IR / forensics / trend analysis — Nitter / Reddit JSON API workarounds beat the "browser_fetch → login wall → fail" dead end.
- **Connect a database to any session** — Postgres / MySQL / SQLite / MSSQL / Oracle / DuckDB via SQLAlchemy. Two-step Test → Connect modal. Destructive SQL (`DELETE` / `DROP` / `TRUNCATE` / `UPDATE`) is hard-blocked on every code path — defence in depth.
- **Per-session workspace isolation** — each new session gets its own `agent-workspace/sessions/<id>/`; files from one session never bleed into another. Toggleable in Settings.
- **Self-growing skills** — recurring goal patterns are auto-detected via ChromaDB embedding similarity (no model self-flagging needed); when N+ similar past goals are found above the similarity floor, OpenTeddy synthesises a skill name + description and asks Claude to generate the Python function. Tunable via Settings → "Skill auto-detect" knobs (default: 3 repeats at 0.75 similarity).
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
- **Drive OpenTeddy from Telegram** — toggle on a single bot token + chat-ID whitelist in Settings → Notification Credentials and your agent is reachable from your phone over Telegram: send any text as a goal, get live "⚙️ Subtask 3/5 · 🎯 0.85 · 💰 $0.043" progress that edits in place, a final summary, and `📎 Files produced` with text artifacts inlined / binary artifacts sent as tap-to-download attachments. Auto-approves high-risk tools so you don't have to leave Telegram for routine work — but a hard denylist (`rm`, `rmdir`, `DROP TABLE`, `TRUNCATE TABLE`, `DELETE FROM`, `mkfs`, `dd if=…/of=/dev/…`, …) hard-blocks destructive actions regardless. Commands: `/start`, `/help`, `/cancel`, `/new`. See [Remote Access](#remote-access-phone--telegram--tailscale).
- **Phone-friendly web UI over Tailscale** — `./run.sh --host 0.0.0.0` + your tailnet means the dashboard works from your phone's browser too. Sessions / chat-mode / artifact previews all responsive; mobile header collapses the session controls into a ⋯ kebab. No port-forwarding, no nginx, no public DNS — just install the Tailscale app on your phone and hit `http://<your-machine>:8000`.
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

`./run.sh` auto-activates `.venv`, pings the Ollama daemon, then runs `uvicorn main:app --reload` so editing source hot-reloads the backend. Common flags:

```bash
./run.sh                    # local-only on 127.0.0.1:8000 (the safe default)
./run.sh --open             # also opens http://localhost:8000 in your browser
./run.sh --port 8001        # bind a different port
./run.sh --host 0.0.0.0     # ⚠ expose to LAN / Tailscale / other machines
./run.sh --no-reload        # production-style — don't watch for file changes
./run.sh --help             # full flag list
```

> ⚠️ **`--host 0.0.0.0` opens the agent to every machine that can reach the port.** The agent has `shell_exec_write` / `delete_file` and other powerful tools. Only use `0.0.0.0` when you trust every device on that network — a private home LAN, a Tailscale tailnet, or a server behind a real firewall. For public servers, put it behind nginx / Caddy / Cloudflare Tunnel with auth. **For "I want to use OpenTeddy from my phone", the recommended setup is `--host 0.0.0.0` + Tailscale — see [Remote Access](#remote-access-phone--telegram--tailscale).**

Customisation flags for the **installer**: `--dir <path>`, `--force`, `--skip-models`. See `./install.sh --help`.

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

## Linux Server Setup (DGX / Jetson / headless fleet) *(v1.1.5)*

If you're deploying OpenTeddy on a Linux server (NVIDIA DGX Spark, Jetson, Raspberry Pi 5, a rack box, anywhere without a permanent monitor), there are extra moving parts beyond `uvicorn main:app`:

- a **Chromium-based browser** running as a systemd service so OpenTeddy can scrape login-gated sites (X / Twitter, LinkedIn, paid SaaS, corp wikis) via the [`chrome_attached_tool`](tools/chrome_attached_tool.py) suite
- a **one-time login** so the headless browser carries your real authenticated session — `x_search('黴菌', top_n=10)` is meaningless if the underlying browser is signed out
- a **profile dir + systemd unit** so the setup survives reboots

Three bundled scripts under [`scripts/`](scripts/) handle the entire flow:

| Script | When to run |
|---|---|
| `scripts/quickstart.sh --login` | **First time only** on a fresh node. Sets up everything below in one command. |
| `scripts/setup-edge-cdp.sh` | Manually invoked by quickstart. Installs Brave (ARM64) or Microsoft Edge (amd64) + writes the systemd unit. |
| `scripts/login-helper.sh` | Whenever a site's cookies expire (X ~30 days, LinkedIn ~90). Just opens a GUI browser for you to re-login. |

### One-command first-time setup

On a GUI-capable session (AnyDesk, physical monitor, `ssh -X`, GNOME/KDE):

```bash
git clone https://github.com/m31527/OpenTeddy
cd OpenTeddy
bash scripts/quickstart.sh --login
```

This will:

1. **Install the CDP browser stack** — auto-detects host arch:
   - `aarch64` / `arm64` → **Brave Browser** (official ARM64 apt repo, no sandbox issues)
   - `x86_64` → **Microsoft Edge** (official amd64 apt repo)
2. **Write a systemd unit** (`openteddy-cdp.service`) that keeps the browser running headless on `127.0.0.1:9222` with `Restart=always`. Survives reboots.
3. **Open a GUI browser window** for you to log in to X / LinkedIn / whatever. Close the window when done — the script automatically switches back to headless mode, cookies persisting in the profile dir.
4. **Create the Python venv** + `pip install -r requirements.txt`.
5. **Start the OpenTeddy backend** detached on `:8000` + healthcheck-verify it.

After this, `http://<host>:8000` is a fully working OpenTeddy. Chat "整理 X 上最近討論黴菌的熱門推文 top 10" and the [`x_search`](tools/chrome_attached_tool.py) tool will pull real tweets through your logged-in session.

### What if I'm pure-SSH without a GUI?

You have two clean options:

- **`ssh -X admin@<host>`** — Brave's window forwards to your local laptop's screen. Works as long as you have X11 on your laptop (Linux has it natively, macOS needs [XQuartz](https://www.xquartz.org/)).
- **`scripts/setup-novnc-login.sh`** — installs Xvfb + noVNC so you can log in via a web browser:
  ```bash
  sudo bash scripts/setup-novnc-login.sh
  sudo systemctl start openteddy-novnc-login.service
  # On your laptop:
  ssh -L 6080:localhost:6080 admin@<host>
  # Then open http://localhost:6080/vnc.html in any browser
  # Log in via the Brave window that appears, close the tab, then:
  sudo systemctl stop openteddy-novnc-login.service
  ```
  noVNC binds to `127.0.0.1` only by default — the SSH tunnel is the auth + encryption layer. To expose more widely (Tailscale mesh, etc.), pass `OPENTEDDY_NOVNC_BIND=0.0.0.0` to the setup script; the install summary explains the security trade-offs.

### Re-login when cookies expire

You'll know cookies have expired when `x_search` starts returning empty `posts: []` or your scheduled trend-tracking task starts coming back blank. Just run:

```bash
bash scripts/login-helper.sh
```

It pauses the headless service, opens a GUI browser pointed at `https://x.com/login`, waits for you to log in and close the window, then automatically restarts the headless service. Takes about 2 minutes.

### macOS desktop setup (same pattern, different OS)

If you're running OpenTeddy on a Mac (Apple Silicon or Intel) and want
the same browser-scraping capability without dropping into Terminal every
time, the macOS counterpart of the above is in `scripts/setup-mac-chrome.sh`
+ `scripts/login-mac-helper.sh`:

```bash
# First-time setup — installs a LaunchAgent that keeps Chrome (or Brave /
# Edge / Chromium if Chrome isn't installed) running headless on
# 127.0.0.1:9222 across reboots. Idempotent.
bash scripts/setup-mac-chrome.sh

# Whenever you need to log in to a site (X / Threads / LinkedIn / etc.):
bash scripts/login-mac-helper.sh
# → temporarily swaps in a headful Chrome window pointed at the same
#   profile; log in, close the window, the LaunchAgent restarts the
#   headless instance automatically.

# Uninstall:
bash scripts/setup-mac-chrome.sh --uninstall
```

Key difference from Linux: the macOS setup uses a SEPARATE Chrome profile
(`~/Library/Application Support/OpenTeddy/Chrome-CDP`) instead of sharing
your day-to-day profile. That way OpenTeddy's scraping Chrome runs
alongside the Chrome window you have open for normal browsing — no
killing your existing tabs every time. Trade-off: you have to log in to
scraping sites once inside the OpenTeddy profile (via
`login-mac-helper.sh`), separately from your normal Chrome.

### Fleet deployment (5-10 nodes)

The same three scripts work as an Ansible playbook payload — each node gets identical setup, then each operator-trusted node gets its own login once. Pattern that we use on NVIDIA DGX Spark fleets:

```bash
# One-time per node (e.g. via Ansible / cloud-init)
ansible all -m shell -a "git clone https://github.com/m31527/OpenTeddy /opt/openteddy"
ansible all -m shell -a "bash /opt/openteddy/scripts/quickstart.sh"

# Then on each node, an operator logs in via AnyDesk / VNC + runs:
#   bash /opt/openteddy/scripts/login-helper.sh
```

Each node owns its own browser profile + cookies; no cross-node cookie sync needed. If you want fleet-wide cookie sharing (one login, all nodes inherit), drop a `storage_state.json` (Playwright format) at `/var/lib/openteddy/storage_state.json` — `chrome_attached_tool` auto-injects it on every attach. See [`scripts/capture-edge-state.md`](scripts/capture-edge-state.md) for the capture recipe.

## Remote Access (phone / Telegram / Tailscale)

Two complementary ways to reach your OpenTeddy instance away from the machine
it runs on. Both work against the same server and the same sessions — you
can start a goal in Telegram on the train and finish reading the artifact
output in the desktop web UI when you're back home.

### A. Web UI from your phone via Tailscale

The simplest "let me check on the agent from anywhere" setup. Zero
port-forwarding, no DNS, no nginx.

1. **On the server** (the machine running OpenTeddy):
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ./run.sh --host 0.0.0.0
   ```
   `--host 0.0.0.0` makes uvicorn bind to all interfaces (including the
   tailnet one). Tailscale itself is what restricts who can actually reach
   the port — only devices on your tailnet.

2. **On your phone**: install the Tailscale app from the App Store / Play
   Store, sign in with the same account, turn on the VPN toggle.

3. **Open the browser** and hit `http://<machine-name>:8000` (or the
   tailnet IP from `tailscale status`). The web UI loads with the same
   sessions, the same artifact chips, the same WebSocket live stream. On
   a phone-width screen the session header collapses memory / privacy /
   export controls into a ⋯ kebab and the mode switcher reflows.

> ⚠️ **Why Tailscale instead of just `--host 0.0.0.0` on the LAN?** Plain
> LAN exposure means anyone on your WiFi (including guest devices) can
> drive your agent — and the agent has `shell_exec_write` and friends.
> Tailscale ACLs let you keep the port reachable from only the devices
> you explicitly approve. If you do want plain LAN, only do it on a
> private home network you fully control.

### B. Bidirectional Telegram bot

Send a goal from anywhere, get the result pushed back to the same chat.
Live progress updates edit a single message in place — no spam. Built for
self-hosted servers that stay running 24/7 (Mac mini, NUC, home Linux
box) — long-polling stops when the server stops, so this is a worse fit
for the desktop app that you close and reopen all day.

#### 1. Create a bot

Open Telegram, talk to **@BotFather**, send `/newbot`, follow the prompts.
Save the bot token (looks like `123456:ABC-DEF1234...`).

#### 2. Find your numeric chat-ID

Send any message to **@userinfobot**. It replies with your numeric `id`
(e.g. `987654321`). For a group: send a message in the group, then
forward it to **@userinfobot** — it shows the group's chat-ID (negative
number, e.g. `-1001234567890`).

#### 3. Start a chat with your bot

Search for your bot's `@username` in Telegram and tap **Start** (or send
`/start`). This is the single most-missed step — without it, Telegram's
"bot can't message you out of the blue" rule kicks in and every
outbound test pings back `chat not found`.

#### 4. Configure OpenTeddy

In **Settings → Notification Credentials**:

| Field | Value |
|---|---|
| Bot Token | the token from BotFather |
| Default Chat ID | your numeric chat-ID (lets the `telegram_send` tool work) |
| **Test ping** button | click — expect ✓ + a "🐻 OpenTeddy ping" message in Telegram |
| **Enable inbound polling** | ✅ check |
| Chat-ID whitelist | your chat-ID(s), comma-separated |

Save. **Restart the server** (hot-reload of the toggle is on the
backlog — for now `./run.sh` must be restarted to start the polling
loop). On boot you should see in the log:

```
[INFO] telegram_bridge: Telegram inbound bridge started — polling with 1-id whitelist.
```

If you instead see `Telegram inbound bridge NOT started: …` the message
spells out exactly which field needs another look.

#### 5. Drive the agent from Telegram

| You send | What happens |
|---|---|
| any text | run as a goal in this chat's bound session; reply with the result |
| `/start` | confirm you're connected |
| `/help` | command list |
| `/cancel` | abort the currently-running task |
| `/new` | start a fresh session (old one stays in history) |

The agent's reply includes:

- **Status line** — `✅ Completed · 12.4s · 3 subtasks`
- **Summary text** — the orchestrator's final summary, capped at ~3500 chars
- **`📎 Files produced`** — every artifact (incl. shell-redirect outputs
  caught by the post-subtask workspace scanner) with size + emitting tool
- **Inline content** — text artifacts under 3 KB sent as a follow-up
  message; larger / binary artifacts uploaded via Telegram `sendDocument`
  so you can tap-to-download from inside the chat.

#### Safety model

- **Hard whitelist**: messages from any non-whitelisted chat are silently
  dropped (no probe signal). Empty whitelist = inbound refuses to start
  even if the toggle is on.
- **Auto-approve high-risk tools**: the whitelisted chat-ID is itself the
  consent signal — `shell_exec_write`, `python_exec`, `file_write` and
  friends run without web-UI approval prompts that nobody would see.
- **Hard denylist on destructive ops**: regardless of approval, the tool
  registry blocks anything matching `rm` / `rmdir` / `unlink` / `git rm`
  / `shred`, SQL `DROP TABLE` / `DROP DATABASE` / `TRUNCATE TABLE`
  / `DELETE FROM`, system-level `mkfs` / `dd if=…/of=/dev/…`
  / `> /dev/sd[a-z]` / `fdisk` / `format X:` / recursive `chmod 0…`,
  plus any tool whose name matches `*delete*` / `*remove*` / `*drop_table*`
  / `*truncate*` / `*wipe*` / `*purge*`. The agent's reply explains the
  block and points at the web UI for interactive approval.
- **10-min hard timeout**: a single Telegram-driven run is wrapped in
  `asyncio.wait_for(timeout=600)` so a hung Ollama call or tool deadlock
  can't freeze the chat — the bot replies `⌛ Task ran longer than 10
  min and was force-cancelled` instead of going silent.

#### Diagnostics

```bash
curl -s http://<server>:8000/admin/telegram/status | jq
```

Returns the bridge's runtime state — `running`, `inbound_enabled`,
`token_set`, the parsed whitelist, the most recent silently-dropped
`chat_id` (the single fastest answer to "why isn't my bot replying?"),
and any in-flight chats. Token is never returned, only a boolean flag.

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
| `GET`  | `/admin/telegram/status` | Inbound bridge runtime snapshot (running, whitelist, last-dropped chat_id, in-flight chats). Safe to expose — never returns the bot token. |
| `POST` | `/settings/telegram/test` | One-shot "OpenTeddy is connected" message to the default chat — friendly-error remapping translates Telegram's terse codes into 30-second fix steps. |
| `GET`  | `/sessions/{id}/export` | Download a single-JSON dump of the session (metadata + tasks + subtasks + memory + DB connection with password masked). Used by the chat header's ⋯ kebab → 📥 Export. |
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

OpenTeddy learns by spotting **patterns in your usage**, not by asking the
local model to introspect about itself. The mechanism:

1. **Embedding-based pattern detection** — after each successful task,
   the orchestrator searches ChromaDB for past `task_result` memories
   whose embedding is semantically close to the just-finished goal.
   When ≥ `SKILL_AUTO_DETECT_MIN_REPEATS` (default **3**) past goals
   score ≥ `SKILL_AUTO_DETECT_SIMILARITY` (default **0.75**) — that's a
   recurring pattern.
2. **Cluster → skill synthesis** — the recurring goals are bundled into
   a small prompt to your configured orchestrator LLM (Gemma locally,
   or your cloud provider in Cloud mode), which returns a JSON
   `{name, description}` capturing the reusable function.
3. **Code generation** — `SkillFactory.generate_skill(name, description)`
   asks Claude (or whichever cloud LLM is configured) to write the
   `async def run(input_data: dict) -> str` function and saves it to
   `skills/<name>.py`.
4. **Promotion** — the skill starts in `TESTING`. After
   `SKILL_PROMOTION_THRESHOLD` (default 5) successful invocations it's
   promoted to `ACTIVE`.
5. **Future task matching** — future goals that match an `ACTIVE`
   skill at ≥ `SKILL_MATCH_THRESHOLD` (default 0.4) invoke it
   directly, skipping the LLM tool-call round entirely.

### Why this design

The original mechanism asked the executor LLM to set
`skill_needed`/`skill_description` in its JSON output. Empirically, 2-3B
parameter models almost never produce that kind of metacognitive
self-flag — verified on a real install with > 100 tasks and zero
auto-generated skills. The embedding approach moves the "is this
recurring" judgment from the model's introspection (unreliable) to a
deterministic similarity check against memory (reliable).

Tunable knobs live in Settings → Parameter Settings, or via the
`SKILL_AUTO_DETECT_*` env vars in `.env`. Set `min_repeats=0` to
disable auto-detection entirely (skills can still be created manually
via `POST /skills/generate?name=…&description=…`).

## ☕ Buy Me a Coffee

OpenTeddy is a one-person side project. Hosting, cloud-LLM API testing,
the macOS signing + notarisation pipeline, and the time it takes to keep
shipping new tools all cost real money and weekends each month. If this
project saves you time, please consider chipping in:

[**☕ Buy Me a Coffee →**](https://openteddy-app.lemonsqueezy.com/checkout/buy/103ae6c2-36cf-48e1-aefc-71faca140657)

No pricing tiers, no feature gates, no "premium". Everything in this repo
stays MIT-licensed and free forever — the coffee just buys me a few extra
hours to keep adding tools, fixing planner edge cases, and writing the
docs nobody else will write.

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
| `SKILL_PROMOTION_THRESHOLD` | `5` | Successes needed to promote a `TESTING` skill to `ACTIVE`. |
| `SKILL_AUTO_DETECT_MIN_REPEATS` | `3` | Min number of past goals that must match the current goal (above the similarity floor) before OpenTeddy synthesises a new skill. Set to `0` to disable auto-detection entirely. |
| `SKILL_AUTO_DETECT_SIMILARITY` | `0.75` | Cosine-similarity floor (0.0-1.0) for counting a past task as "recurring". Calibrated against ChromaDB's default MiniLM embedder; bump to ~0.9 if you see weird skills getting generated, drop to ~0.65 if expected patterns aren't being caught. |
| `SKILL_MATCH_THRESHOLD` | `0.4` | Min similarity to match an existing `ACTIVE` skill against a new goal — skips the LLM round entirely on match. |
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

### Notification credentials

All blank by default — the relevant tools / bridges report a clear
"not configured" error pointing at Settings, so a stock install never
silently does the wrong thing. These can all be edited from
**Settings → Notification Credentials** at runtime; the env-var form
is just for headless / Docker-compose installs.

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | — | Bot token from @BotFather. Required for both outbound `telegram_send` and inbound polling. |
| `TELEGRAM_DEFAULT_CHAT_ID` | — | Optional. When set, `telegram_send` and `/settings/telegram/test` can omit `chat_id` and send here by default. |
| `TELEGRAM_INBOUND_ENABLED` | `false` | Master toggle for the long-polling Telegram→OpenTeddy bridge (see [Remote Access → Bidirectional Telegram bot](#b-bidirectional-telegram-bot)). |
| `TELEGRAM_INBOUND_WHITELIST` | — | Comma-separated chat-IDs allowed to drive the agent (numeric for users / groups, `@channelname` for public channels). **Empty = inbound refuses to start even if the toggle is on** — we don't run open bots. |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` | — | Used by the `email_send` tool. `SMTP_PORT` defaults to `587`. |
| `WEBHOOK_SECRET` | — | Optional shared-secret for `POST /webhooks/{session_id}`. Empty = endpoint is open to anyone on the network (UI warns when this is the case). |

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
| **macOS** (Apple Silicon) | ✅ Native desktop `.dmg` (signed + notarized) + OSS web | Primary development target. |
| **Linux** (x86_64) | ✅ Native desktop `.AppImage` / `.deb` *(NEW v1.0.3)* + OSS web | Built on Ubuntu 22.04 CI; AppImage works on any glibc 2.34+ distro. |
| **Windows (native)** | ⚠️ Partial — use WSL2 if possible | See caveats below. No native desktop installer yet (roadmap). |
| **Windows (WSL2)** | ✅ Fully supported (OSS web) | Behaves like Linux. Recommended on Windows. |

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

OpenTeddy itself is **MIT**.

Third-party content bundled in this repo:

| Bundled artifact | Source | Upstream license |
|---|---|---|
| `cyber_skills/index.json` (the indexed 755-workflow catalogue) | [mukul975/Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills) (754 entries) + [mvanhorn/last30days-skill](https://github.com/mvanhorn/last30days-skill) (1 entry) | Apache 2.0 + MIT |
| `tools/doc_to_markdown.py` wrapper around [microsoft/markitdown](https://github.com/microsoft/markitdown) | upstream PyPI package | Apache 2.0 |

`cyber_skills/index.json` is a **derivative work** — see
`cyber_skills/README.md` for attribution details. Every indexed entry
carries `source_repo` + `upstream_url` fields so any single workflow
can be traced back to its origin. Refer to the linked upstream repos
for the full license text and NOTICE files where applicable.

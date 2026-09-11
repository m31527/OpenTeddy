# openteddy.net — homepage v2 (handoff)

Source: *OpenTeddy Discovery & Adoption Audit* (Sept 2026). The site's source
lives outside this repo, so this file is the spec: structure, copy in EN and
繁中, the three demos to record, and how we'll know it worked. The README was
rewritten to the same structure in this commit — keep the two in sync; the
README is the source of truth for commands.

Positioning phrase — use it verbatim everywhere (site, README, release posts,
demo captions):

> **local AI agent · autonomous work · self-growing skills · hybrid escalation**

---

## What to change and why (one screen)

| Today | Problem the audit found | v2 |
|---|---|---|
| Title: *"Free, Claude-like AI Agent that runs locally"* | "Claude-like" borrows someone else's identity and invites a comparison we lose; the outcome is missing | Outcome-led headline (below) |
| Hero → 12 feature cards (Local Models, Skills, Claude Escalation, Local-First, …) | Capabilities compete with each other; a first-time developer can't say what they'd *accomplish* | One job-to-be-done, then ONE proof, then three reasons |
| "Up and running in minutes" = installation | Installation is not the destination; the first finished artifact is | "Your first finished task" with the exact command and what appears |
| README said "at minimum set ANTHROPIC_API_KEY" while the site says local-first | Contradiction → doubt for exactly the developer we want | Two explicit paths with requirements and *what leaves the machine* |
| Meta description ends with "Lifetime $99" | Pricing in the first sentence, before value | Pricing moves to the bottom of the page (desktop app section) |

---

## Page structure

### 1. Hero (outcome-led)

**EN**

> # A local AI agent that does the work — and gets better at repeated tasks.
> Runs on your machine with Ollama. Escalates to a cloud model only when it actually gets stuck. Turns the work you repeat into reusable skills.
>
> [Try it locally — one command]   [Watch the 45-second proof]
>
> `curl -fsSL https://openteddy.net/install | bash`

**繁中**

> # 在你機器上把事情做完的 AI agent —— 而且重複做的事會越做越熟。
> 用 Ollama 在本機執行。只有真的卡住時才升級到雲端模型。重複的工作會變成可重用的技能。
>
> [一行指令，本機試用]   [看 45 秒證明]

`<title>`: **OpenTeddy — a local AI agent that does the work**
`<meta description>`: *Run an autonomous AI agent on your own machine with Ollama. It finishes real tasks — reports, code, data — escalates to a cloud model only when stuck, and turns repeated work into reusable skills. Open source.*

Rules: one headline, one sentence, one primary CTA, one secondary CTA. No feature grid above the fold.

### 2. One proof (30–60 s, autoplay muted, captions)

Record **Demo A** (below). Caption under the video:

> *One sentence in, one finished report out — planned, executed and verified on a laptop, no API key.*
> *一句話進去，一份完成的報告出來 —— 在筆電上規劃、執行、驗證，沒有 API key。*

### 3. Three reasons to care

| Local control | Lower recurring cost | Reusable skills |
|---|---|---|
| Planning and execution run on your hardware. A local-only session never calls a cloud API. | $0 per token by default. A cloud model runs only when a step genuinely fails — and the Usage tab shows what GPT-4 would have charged. | Work you repeat becomes a plain Python skill; the next time costs no reasoning at all. |
| 規劃與執行都在你的硬體上。純本地 session 絕不呼叫雲端 API。 | 預設每個 token $0。只有某一步真的失敗才叫雲端模型 —— Usage 頁會告訴你 GPT-4 要收多少。 | 重複的工作變成純 Python 技能，下一次完全不用推理。 |

### 4. Pick your path (requirements made explicit)

Three cards. Each states *what you need* and *what leaves your machine* — the audit's documentation-consistency fix lives here.

| Try locally | Use hybrid models | Developer docs |
|---|---|---|
| Python 3.10+ · Ollama · 8 GB RAM (16 GB comfortable) | The above + **one** cloud key (Anthropic / OpenAI / Gemini / Deepseek / OpenRouter) | git + venv, or Docker |
| **Nothing leaves your machine.** | Only escalated subtasks and skill-generation prompts leave; local-only sessions still send nothing. | — |
| Full agent loop, tools, memory, dashboard, CLI, schedules, Telegram. No escalation, no skill generation. | + auto-escalation safety net, + self-growing skills | README → Quick Start, Task API, CLI |
| `curl -fsSL https://openteddy.net/install \| bash` | Settings → Cloud LLM Provider | github.com/m31527/OpenTeddy |

### 5. Three proof workflows (input → agent steps → artifact)

Short cards, each with the command, three step icons, and a thumbnail of the result. Link each to its demo.

1. **Report from data** — `./openteddy run "Analyze ~/OpenTeddy/examples/sales_sample.csv: monthly revenue trend, top 3 products, one chart. Save as HTML."` → csv_describe → python_exec → render_chart_report → **HTML report with a chart**
2. **A coding task** — `./openteddy run "Create a Python CLI that converts CSV to JSON (with --pretty) and a pytest test. Run the tests." --mode code` → write_file → shell (pytest) → **files + passing tests**
3. **Repeated work becomes a skill** — run the report task three times → pattern detected → skill synthesised and tested → `openteddy skill run <name>` → **same result, zero reasoning**

### 6. Community proof

- Recent releases (pull from GitHub Releases API — the site should not hand-maintain this)
- "277 commits · active releases" style counters, auto-updated
- Example workflows contributed by users (start with the three above; replace with real user ones from the 30-day loop)
- Contribution paths: report a first-task issue · share a skill · share a workflow

### 7. Desktop app + pricing (bottom)

Keep the current macOS/Linux desktop section and Lifetime $99 here, after the value is established. Copy unchanged.

---

## The three demos to record (30–60 s each)

Rules: real terminal, real dashboard, no cuts inside a step, captions for every step, end on the artifact. Record on a laptop with the default models so the timing is honest.

**Demo A — Report from data (the hero proof)**
1. `./openteddy run "Analyze ~/OpenTeddy/examples/sales_sample.csv: monthly revenue trend, top 3 products, one chart. Save as HTML."`
2. Show the terminal trail: plan → tool calls → 📎 path → `✓ Done (completed)`
3. Open the HTML report; hover the chart.
4. Last frame: the dashboard session with the 📎 chip. Caption: *no API key was used.*

**Demo B — Coding task**
1. `./openteddy run "Create a Python CLI that converts a CSV file to JSON (with a --pretty flag) and a pytest test for it. Run the tests." --mode code`
2. Show one `y/N` approval prompt being answered — this is a feature, show it.
3. End on the pytest output quoted in the summary and the files in the workspace.

**Demo C — Repeated work becomes a skill** (hybrid; needs a cloud key)
1. Run the report task on three different CSVs (speed up between runs).
2. `./openteddy skill list` → the new skill, TESTING.
3. `./openteddy skill run <name> --input '{"path": "..."}'` → result in seconds. Caption: *the third time, it's a function.*

---

## 30-day measurement (what "it worked" means)

The audit's principle: optimise for the *right* developer completing one useful task, not for attention.

| Week | Do | Success signal |
|---|---|---|
| 1 | Ship hero + Demo A + path cards | A new visitor can say what OpenTeddy does and who it's for after a 10-second scan (ask 5 people) |
| 2 | Ship Demos B and C + proof cards | Visitors reach the install command without opening the architecture section (scroll-depth / click on primary CTA) |
| 3 | Publish the same idea in developer-native form: README (done), a release post, a short technical post ("What if your local AI could escalate only when it actually gets stuck?") in 2–3 communities | Installs → **first completed task**. We have no telemetry by design, so measure: installer downloads, `examples/sales_sample.csv` mentions in issues/discussions, stars, and a "my first task" discussion thread |
| 4 | Ask the first 20 users which workflow they ran, what confused them, what they'd show a colleague | Real user language and workflows to replace the placeholder proof cards |

Add a GitHub Discussions category **"My first task"** and link it from the README's first-task section — that is the cheapest source of both proof and language.

---

## Do not

- Do not put the feature grid back above the fold.
- Do not say "Claude-like" in the hero; say what it does.
- Do not lead with pricing.
- Do not claim timings we haven't recorded; the demo *is* the timing.

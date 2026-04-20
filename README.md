# 🐻 OpenTeddy

**A self-growing multi-agent system built on Gemma 4 + Qwen 3 + Claude**

OpenTeddy orchestrates a trio of AI agents to autonomously plan, execute, and
improve over time — automatically generating Python "skills" to handle recurring
tasks faster on future runs.

---

## Architecture

```
User Goal
   │
   ▼
┌──────────────────────────────────────────┐
│  Orchestrator  (Gemma 4 via Ollama)       │
│  • Decomposes goal into ordered SubTasks  │
│  • Drives execution loop                  │
└───────────────┬──────────────────────────┘
                │ SubTasks
                ▼
┌──────────────────────────────────────────┐
│  Executor  (Qwen 3 via Ollama)            │
│  • Runs matching Skill if available       │
│  • Falls back to LLM inference            │
│  • Reports confidence score               │
└───────────────┬──────────────────────────┘
       low conf │ or failure
                ▼
┌──────────────────────────────────────────┐
│  Escalation Agent  (Claude via API)       │
│  • Resolves hard subtasks                 │
│  • Synthesises final summary              │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│  Skill Factory  (Claude via API)          │
│  • Generates new Python skills on demand  │
│  • Promotes skills after N successes      │
│  • Saves skills to disk + SQLite DB       │
└──────────────────────────────────────────┘
```

## File Structure

```
OpenTeddy/
├── config.py          # Config via .env / environment variables
├── models.py          # Pydantic models + SQLite schema
├── tracker.py         # Async SQLite persistence (aiosqlite)
├── skill_factory.py   # Claude-powered skill generation & loader
├── executor.py        # Qwen executor agent
├── escalation.py      # Claude escalation agent
├── orchestrator.py    # Gemma orchestrator
├── main.py            # FastAPI server + CLI entry point
├── skills/            # Auto-generated skill .py files
├── static/index.html  # Web dashboard
├── .env.example       # Environment variable template
└── .gitignore
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with:
  ```bash
  ollama pull gemma3:4b
  ollama pull qwen2.5:3b
  ```
- Anthropic API key (for escalation and skill generation)

### 2. Install dependencies

```bash
cd OpenTeddy
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install fastapi uvicorn[standard] aiosqlite httpx anthropic pydantic python-dotenv
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
| `GET`  | `/tasks` | List recent tasks |
| `GET`  | `/skills` | List all skills |
| `POST` | `/skills/generate?name=…&description=…` | Manually create a skill |
| `GET`  | `/health` | Health check |

### Example request

```bash
curl -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Summarise the key benefits of async Python", "priority": 7}'
```

## Self-Growth Mechanism

1. When Qwen executes a subtask it suggests a **skill name** if a reusable
   function would have helped.
2. The Executor calls `SkillFactory.generate_skill()` in the background.
3. Claude writes the skill as an `async def run(input_data: dict) -> str`
   function and saves it to `skills/<name>.py`.
4. The skill starts in **TESTING** status. After `SKILL_PROMOTION_THRESHOLD`
   successful invocations it is promoted to **ACTIVE**.
5. Future tasks automatically match and invoke active skills — no LLM call needed.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Claude model for escalation |
| `GEMMA_BASE_URL` | `http://localhost:11434` | Ollama base URL for Gemma |
| `GEMMA_MODEL` | `gemma3:4b` | Gemma model tag |
| `QWEN_BASE_URL` | `http://localhost:11434` | Ollama base URL for Qwen |
| `QWEN_MODEL` | `qwen2.5:3b` | Qwen model tag |
| `DB_PATH` | `openteddy.db` | SQLite database path |
| `SKILLS_DIR` | `skills` | Directory for skill files |
| `ESCALATION_THRESHOLD` | `0.6` | Min Qwen confidence before escalation |
| `ESCALATION_FAILURE_LIMIT` | `3` | Max consecutive failures before abort |
| `SKILL_PROMOTION_THRESHOLD` | `5` | Successes needed to promote a skill |

## License

MIT

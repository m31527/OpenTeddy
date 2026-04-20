#!/usr/bin/env bash
# OpenTeddy — git init + GitHub push
# Run this once from your terminal:
#   chmod +x setup_git_and_push.sh && ./setup_git_and_push.sh

set -e
cd "$(dirname "$0")"

echo "🐻 OpenTeddy — git setup"

# 1. Clean up any broken .git state from the sandbox
rm -rf .git .git_new

# 2. Fresh git init
git init
git branch -m main
git config user.email "m31527@gmail.com"
git config user.name "Sean"

# 3. Stage and commit everything
git add -A
git commit -m "feat: initial OpenTeddy multi-agent system

- Gemma 4 Orchestrator: decomposes goals into ordered subtasks
- Qwen 3 Executor: runs skills or falls back to LLM inference
- Claude Escalation: resolves low-confidence/failed subtasks
- Skill Factory: Claude-generated async Python skills, auto-promoted
- FastAPI server with /run, /tasks, /skills endpoints
- Web dashboard (static/index.html)
- aiosqlite persistence with full task/subtask/skill tracking
- .env-based configuration"

# 4. Create public GitHub repo and push
gh repo create m31527/OpenTeddy \
  --public \
  --description "Self-growing multi-agent system: Gemma Orchestrator + Qwen Executor + Claude Escalation + Skill Factory" \
  --source=. \
  --remote=origin \
  --push

echo ""
echo "✅ Done!  https://github.com/m31527/OpenTeddy"

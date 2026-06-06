# OpenTeddy Benchmark Harness

Measure "did this change actually help?" with numbers, not vibes.

The next-major-v2 line touches enough subsystems (model routing,
parsing, prompts, skills) that a regression hidden inside a fast-
ping chat run wouldn't be obvious to anyone reading the diff. This
folder is the answer: a small set of representative tasks + a runner
that drives them against any branch, captures metrics, and diffs the
JSON so we can say things like *"phase 3a markitdown landed pdf
parsing from 12.1 s to 1.4 s with no chat-mode regression"* rather
than *"feels faster"*.

## Layout

```
tests/bench/
├── README.md             # this file
├── run.py                # task runner — hits the running server,
│                         # captures timing/tokens/subtasks/success,
│                         # writes JSON
├── compare.py            # diff two JSON output files, render a
│                         # human-readable table to stdout
├── golden_tasks.yaml     # the regression-safety golden set
│                         # (chat, scrape, code, multi-step, sanity)
└── results/              # gitignored; per-run JSON output lives here
```

## Workflow

```bash
# 1. Start the server on the branch you want to measure
./run.sh

# 2. In a separate terminal — capture metrics for THIS branch
python tests/bench/run.py \
    --tasks tests/bench/golden_tasks.yaml \
    --runs 3 \
    --output tests/bench/results/baseline_main.json

# 3. Switch branches, restart server, capture metrics again
git checkout phase3a-markitdown
./run.sh   # restart
python tests/bench/run.py \
    --tasks tests/bench/golden_tasks.yaml \
    --runs 3 \
    --output tests/bench/results/candidate_phase3a.json

# 4. Compare the two
python tests/bench/compare.py \
    tests/bench/results/baseline_main.json \
    tests/bench/results/candidate_phase3a.json
```

The output looks like:

```
=== main vs phase3a-markitdown ===

GOLDEN TASKS
  chat_lunar_date           3.2 s →  3.1 s   (-3.1 %)   ✓
  scrape_github_trending    8.4 s →  8.5 s   (+1.2 %)   ✓
  pdf_50pages_parse         12.1 s → 1.4 s   (-88.4 %)  🎉
  ...

regressions: 0
improvements: 4 (≥ 10 %)
unchanged:    6
```

## Metrics captured per task

| Field            | Source                  | Use                          |
|------------------|-------------------------|------------------------------|
| `duration_s`     | wall-clock              | speed regression detection   |
| `tokens_in`      | tracker.get_task_usage  | LLM context cost             |
| `tokens_out`     | tracker.get_task_usage  | LLM generation cost          |
| `cost_usd`       | tracker.get_task_usage  | cloud spend (mixed mode)     |
| `subtask_count`  | tracker.get_subtasks    | planner aggression           |
| `tool_calls`     | counted from subtasks   | executor verbosity           |
| `status`         | tracker.get_task        | completed / failed / running |
| `summary_chars`  | len(task.summary)       | sanity — did we get *some*   |
|                  |                         | output back                  |

Multiple runs (default 3) → median is what we compare so a single
Ollama hiccup doesn't pollute the comparison.

## Adding a new task

Edit `golden_tasks.yaml`. Each entry needs:

```yaml
- id: my_new_task
  goal: "describe the action in plain language"
  mode: chat | code | analytic
  # All of these are OPTIONAL — used for warning emission, never for
  # hard fail. Bench tells you when reality drifts from expectations,
  # doesn't refuse to run.
  expect_max_duration_s: 10
  expect_max_tokens_out: 2000
  expect_max_subtasks: 2
  expect_must_use_tool: browser_fetch   # tool name or null
```

Keep the set lean. 10-20 representative tasks beats 100 redundant ones
for catching regressions, and a faster bench gets run more often.

# RFC: Unified single-model + fewer round-trips

> Status: **draft** · Branch: `feat/unified-single-model`
> Goal: make OpenTeddy noticeably faster by collapsing the two-model,
> many-round-trip design toward a single capable model running a
> tighter loop — the architecture that makes harnesses like Hermes
> feel fast.

## The problem, measured

A non-trivial OpenTeddy task today makes ~12–15 LLM calls across **two
models on (potentially) two engines**:

```
intent classify (Gemma)              1 call
plan            (Gemma)              1 call
execute × N subtasks (Qwen)          N × 2-3 calls   (tool loop)
verify  × N subtasks (Qwen)          N × 1 call      (if verification on)
summary         (Gemma)              1 call
```

Two independent costs compound:
1. **Per-call latency** — bandwidth-bound on DGX-class HW; vLLM only
   helps under concurrency, not single-stream.
2. **Round-trip count** — the agent design. This is the bigger lever
   and the one we control purely in code.

Plus the **two-model split** (Gemma planner on Ollama + Qwen executor on
Ollama/vLLM) means two resident models, two engines, and a model switch
on every plan→execute boundary.

Fast harnesses (Hermes' "model call → tool dispatch → repeat") win by
doing **one model, one loop, few round-trips**. This RFC moves OpenTeddy
toward that.

## Non-goals / invariants

- **Single-machine + desktop unaffected.** Everything here is gated or
  defaulted so existing Ollama/2-model users see no change unless they
  opt in. Same discipline as fleet/vLLM.
- **tools + skills unchanged.** Confirmed model-agnostic: tools dispatch
  via OpenAI-format `tool_calls` through the registry; skills are plain
  `async def run(input_data)` Python. Any tool-calling model works. No
  tool/skill code changes.
- **Cloud escalation, memory, sessions, fleet, Telegram** — untouched.

## Design

### Part A — unify the engine for the planner (remove the Ollama hard-wire)

Today `_gemma_complete` hard-codes `{gemma_base_url}/api/generate`
(Ollama native). The executor already goes through `local_engine`
(Ollama or vLLM). Route the planner/orchestrator calls through the SAME
`local_engine` so:
- the planner can run on vLLM too, and
- planner + executor can share ONE vLLM instance + ONE loaded model.

Implementation: replace `_gemma_complete`'s direct Ollama POST with a
`local_engine` call (it already has the Ollama vs vLLM dialect logic).
`/api/generate` (completion) → `/v1/chat` or `/api/chat` shaped call.

### Part B — one model for both roles (opt-in)

New setting `OPENTEDDY_UNIFIED_MODEL` (default empty = current 2-model
behaviour). When set:
- both planner and executor use that model id,
- on a single vLLM instance,
- Ollama can be stopped entirely (frees memory for a bigger model).

Recommended target: a 27B-class model (e.g. Qwen3.x-27B). Rationale:
a bigger model is reliable enough to need LESS scaffolding (fewer
verify/summary/retry round-trips), so "27B × 3 round-trips" can beat
"7B × 12 round-trips" on BOTH latency and quality. This is a bet to be
measured, not assumed.

Memory check (DGX Spark, 119 GB unified): 27B BF16 ≈ 54 GB + KV cache
≈ 10 GB → fits once Ollama is stopped. The unification is what FREES the
memory (no second resident model).

### Part C — cut round-trips

Ordered by impact / safety:

1. **Verification off by default for big models.** `_verify_deliverable`
   is one extra LLM call per subtask. The README already says to turn it
   off on 35B-class setups. When `OPENTEDDY_UNIFIED_MODEL` is a large
   model, default verification to off (still toggleable). Saves N calls.

2. **Skip summary for deliverable goals.** When the goal is "produce /
   list / extract X" and the executor already emitted the artifact,
   return it directly instead of a separate `_gemma_summarize` call.
   (Complements the v1.1.3 "synthesizer returns deliverables verbatim"
   fix — now we skip the call entirely for that class.) Saves 1 call.

3. **Fold intent-classify into planning.** The intent classifier is a
   separate call just to decide needs_tools / pure_chat. With a capable
   model, the planner can emit that decision as part of its first
   response (a small JSON header), removing the standalone call.
   Saves 1 call. Medium risk — keep the standalone classifier as a
   fallback when the merged path returns malformed output.

4. **Single-pass ReAct lane for light tasks.** Today only pure-chat hits
   the 1-call fast path; everything tool-touching goes through the full
   plan→execute→summary. Add a "single-tool / light" lane: when the task
   looks like a one-or-two-tool job, run ONE ReAct loop (model ↔ tools ↔
   model until done) with no separate planner/summary — exactly the
   Hermes shape. Heavy multi-step tasks keep the full planner.
   Highest impact, highest risk → behind a flag, opt-in first.

### What each part saves (3-subtask example)

| | calls today | after |
|---|---|---|
| intent | 1 | 0 (folded) |
| plan | 1 | 1 |
| execute | 6 | 6 |
| verify | 3 | 0 (off for big model) |
| summary | 1 | 0 (deliverable) |
| **total** | **~12** | **~7** |

Light single-tool tasks drop from ~5 to **~2** via the ReAct lane.

## Rollout / safety

- Each part is independently flagged; none changes default behaviour
  until opted in. A user on Ollama + 2 models + verification sees zero
  change after merge.
- `OPENTEDDY_UNIFIED_MODEL` empty → today's behaviour exactly.
- Big-model verification-off, summary-skip, intent-fold, ReAct-lane each
  get their own guard so we can ship + measure them one at a time.
- Keep the full plan→execute path as the fallback whenever a shortened
  path returns malformed / empty output (degrade to robust, never fail).

## Milestones

| # | Part | Risk |
|---|---|---|
| 1 | A — planner through local_engine (unblocks single-instance vLLM) | med |
| 2 | B — `OPENTEDDY_UNIFIED_MODEL` one-model-both-roles | low |
| 3 | C1+C2 — verify-off-for-big + skip-summary-for-deliverable | low |
| 4 | C3 — fold intent into plan | med |
| 5 | C4 — single-pass ReAct lane | high |

Measure tok/s + wall-clock + task success rate after each milestone
(same task set, unified 27B vs current 7B-2-model) so we keep only the
changes that actually help.

## Open questions

- Exact 27B model id (Qwen3.x-27B variant availability on HF for vLLM).
- Does the ReAct lane reuse `_qwen_execute`'s tool loop, or a new
  trimmed loop? (Prefer reuse.)
- Should intent-fold ship before or after the ReAct lane? (Lean: after —
  the lane reduces how often intent even matters.)

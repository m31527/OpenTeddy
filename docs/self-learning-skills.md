# Self-Learning Skill Architecture

> Branch: `feature/self-learning-skills` · Spec: [skill-plus.md](../skill-plus.md)
> North Star: **Task → Learn → Verify → Execute → Reuse**

OpenTeddy stops treating skills as static plugins and treats them as
**capabilities Teddy can learn**: if it doesn't know how to do something,
it learns the capability (build → test → repair → verify), remembers it,
and reuses it locally without asking the strong model again.

## 1. Architecture audit (what existed → what was done)

| Component | Before this branch | Verdict |
|---|---|---|
| Skill registry | `tracker` skills table + `SkillMetadata` (name/code/version/status/counters) | **Extended** — capabilities, input_keys, permissions, source_type, model_used, test_status, last_used_at, enabled + `skill_versions` history table |
| Skill matching | inline first-keyword-wins substring check in `executor._try_skill` | **Replaced** — `skill_matcher.py`, scored name/capability/description overlap (CJK-aware), threshold = existing `skill_match_threshold` setting |
| Skill builder | `skill_factory` Loop A (generate → syntax → behaviour-test → ≤2 repairs → register ACTIVE) | **Extended** — staged spec→build→test→repair loop via `ensure_skill`, configurable attempts, permission declaration, model routing, version snapshots |
| Test-before-save | shipped earlier (Loop A) | Reused; attempts now `skill_repair_max_attempts` (default 3) |
| Runtime self-heal | Loop A regen + version cap + RETIRED | **Extended** — old code snapshotted to `skill_versions` **before** every overwrite; `rollback_skill()` restores any version (append-only history) |
| Model routing | `llm_mode` (local/mixed/cloud) + per-session `local_only` + escalation | **Formalised** — `model_router.py`: `LOCAL_PREFERRED / STRONG_MODEL_PREFERRED / LOCAL_ONLY`; Local-Only returns a completer that cannot touch the provider object (enforcement by construction) |
| Permissions | tool-level only (risk levels + approval store); skills had none | **New** — `SkillPermissions` schema (filesystem/network/commands/credentials/services), LLM-declared least-privilege, stored per skill, propagated via `RuntimeContext` |
| Execution runtime | in-process exec inline in the factory + a stale-prone module cache | **Extracted** — `skill_runtime.py`: `SkillRuntime` ABC + `NativeRuntime`; skills execute from the DB row (cache bug class removed). Docker/OpenShell runtimes plug in later without touching the builder |
| Observability | plain logs | Structured event names: `skill.search.*`, `skill.build.*`, `skill.test.*`, `skill.repair.*`, `skill.registered`, `skill.execution.*`, `model.route.selected` |
| Tests | none | `tests/test_self_learning_skills.py` — acceptance scenarios A–G, no real LLM calls |

## 2. The loop

```
task ──▶ skill_matcher (scored, threshold configurable)
            │ matched               │ no match
            ▼                       ▼
      invoke_skill          SkillFactory.ensure_skill
            │                 spec (reusable definition, strong model)
            │                 → generate → behaviour-test
            │                 → ≤ skill_repair_max_attempts repairs
            │                 → declare least-privilege permissions
            │                 → register ACTIVE + version snapshot
            ▼                       │
      SkillRuntime.execute ◀────────┘
      (permissions in RuntimeContext, timeout, DB-row code)
            │ raise at runtime
            ▼
      background self-repair (real failing input as regression test)
      → snapshot old version → v+1 … at cap → RETIRED
      → tracker.rollback_skill(name, v) restores any version
```

**Cloud teaches, local executes** (§10): the strong model is paid once to
build a verified skill; every future matching task runs the saved code
locally with zero LLM involvement in the skill itself.

**Local-Only** (§9): `llm_mode=local` or a session privacy flag routes
*everything* — including skill building and repair — to the local model.
The cloud provider object is never invoked (scenario E asserts a
provider spy records zero calls).

## 3. Config

| Setting | Default | Meaning |
|---|---|---|
| `skill_match_threshold` | existing | matcher confidence needed to reuse |
| `OPENTEDDY_SKILL_TEST` (`skill_test_before_register`) | on | behaviour-test before registering |
| `OPENTEDDY_SKILL_REPAIR_MAX` (`skill_repair_max_attempts`) | 3 | LLM repair attempts inside one build |
| `OPENTEDDY_SKILL_REGEN_MAX` (`skill_regen_max_versions`) | 3 | lifetime runtime self-repairs before RETIRED |
| `OPENTEDDY_SKILL_LEARNING` (`skill_learning_enabled`) | **off** | let the executor MINT new skills for unmatched subtasks. The loop is always available programmatically via `ensure_skill`; auto-minting is opt-in because most one-off tasks shouldn't cost a strong-model build |

## 4. Tests

```bash
.venv/bin/python tests/test_self_learning_skills.py
```

| Scenario | Asserts |
|---|---|
| A existing skill | matched + executed, builder call count 0 |
| B new skill | spec→build→test→register ACTIVE→execute |
| C broken generation | error captured → repair → verified; attempts ≤ cap |
| D reuse | second identical task: builder call count 0 |
| E Local Only | zero cloud-provider calls (spy), local model used |
| F permissions | declaration stored; runtime receives it via context |
| G runtime break | self-repair to v2; v1 recoverable; rollback works |

## 5. Known limitations / future work

- **NativeRuntime is not a security sandbox** — generated code runs
  in-process at the historical trust level. The runtime abstraction is
  the seam where DockerRuntime / NVIDIA OpenShell land later.
- Permission declarations are propagated + surfaced, not OS-enforced
  (that is the sandboxed runtime's job).
- Matching is lexical (name/capability/description overlap). Good enough
  with LLM `skill_hint` short-circuiting; semantic matching via the
  existing Chroma store is a natural upgrade.
- Generated "tests" are one behaviour run against an LLM-suggested input,
  not a persisted test suite.
- Auto-minting (`skill_learning_enabled`) is conservative by design.

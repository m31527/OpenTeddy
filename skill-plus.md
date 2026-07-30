# **OpenTeddy — Self-Learning Skill Architecture Implementation Task**

## **Objective**

Implement the next core capability of OpenTeddy:

**OpenTeddy should be able to learn a new skill when it does not already know how to complete a task, validate that skill, save it, and reuse it later.**

Product principles:

**Your AI that learns how you work.**

**Local when possible. Powerful when needed.**

This is NOT a general rewrite of OpenTeddy.

The goal is to add a clean, extensible **Skill Learning Loop** on top of the existing architecture with the smallest reasonable amount of refactoring.

---

# **0. Git Workflow**

Before changing code:

1. Inspect the existing repository.
2. Confirm the current branch and working tree status.
3. Do not overwrite or discard existing uncommitted changes.
4. Create a new branch from the current intended base branch.

Suggested branch:

```bash
git checkout -b feature/self-learning-skills
```

bashIf that branch already exists, create a similar descriptive branch name.

Do not push or merge unless explicitly requested.

---

# **1. First Step: Repository Architecture Audit**

Before implementation, inspect the codebase and identify the current implementation of:

* Skill system
* Tool execution
* Python execution
* Agent/task planning
* LLM provider abstraction
* Ollama/local model handling
* Cloud LLM handling
* Memory
* Prompt management
* Configuration
* Database/storage
* Existing permission/security controls
* Existing retry/error handling
* Existing tests

Do NOT immediately rewrite these systems.

Produce a concise internal implementation plan describing:

```text
Current component
→ Can reuse?
→ Needs extension?
→ Needs refactor?
→ Proposed location/interface
```

textPrefer extending existing abstractions over replacing them.

---

# **2. Core User Experience**

The desired runtime behavior is:

```text
User Request
    ↓
Understand Task
    ↓
Search Existing Skills
    ↓
Can an existing skill solve it?
    │
 ┌──┴──┐
 │     │
YES    NO
 │      │
 │      ↓
 │   Skill Builder
 │      ↓
 │   Generate Skill
 │      ↓
 │   Validate / Test
 │      ↓
 │   Failure?
 │      ↓
 │   Diagnose + Repair
 │      ↓
 │   Re-test
 │      ↓
 │   Verified
 │      ↓
 │   Save Skill
 │      ↓
 └──────┤
        ↓
Execute Skill
        ↓
Validate Result
        ↓
Return Result
```

textThe key product behavior is:

If Teddy does not know how to perform a task, it should try to learn the capability instead of immediately failing.

---

# **3. Priority 1 — Skill Registry**

Create or extend a central Skill Registry.

A skill should have structured metadata similar to:

```typescript
interface SkillDefinition {
  id: string;
  name: string;
  description: string;
  version: string;

  capabilities: string[];

  inputs: SkillInputSchema[];
  outputs: SkillOutputSchema[];

  permissions: SkillPermissions;

  dependencies?: string[];

  sourceType: "builtin" | "generated" | "installed";

  createdBy?: string;
  modelUsed?: string;

  testStatus: "untested" | "passed" | "failed";
  testResults?: SkillTestResult[];

  successCount: number;
  failureCount: number;

  createdAt: string;
  updatedAt: string;
  lastUsedAt?: string;

  enabled: boolean;
}
```

typescriptAdapt naming and types to the current project language and conventions.

Required operations:

```text
registerSkill()
findSkill()
searchSkills()
getSkill()
updateSkill()
disableSkill()
deleteSkill()
executeSkill()
```

textDo not duplicate an existing registry if OpenTeddy already has one.

Extend it instead.

---

# **4. Skill Discovery**

Before generating a new skill, OpenTeddy must search existing skills.

Implement a reusable abstraction such as:

```text
SkillMatcher
```

textInput:

```text
User task
Task intent
Task constraints
Available skills
```

textOutput:

```json
{
  "matched": true,
  "skillId": "...",
  "confidence": 0.91,
  "reason": "..."
}
```

jsonIf confidence is high enough:

```text
reuse existing skill
```

textDo NOT generate a duplicate skill.

If no suitable skill exists:

```text
invoke Skill Builder
```

textStart with a pragmatic matching approach.

It may combine:

* metadata
* keywords
* descriptions
* embeddings if already available
* LLM classification if appropriate

Do not introduce a heavy vector database solely for this feature unless the repository already uses one.

---

# **5. Priority 2 — Skill Builder**

Implement a dedicated Skill Builder abstraction.

Suggested conceptual interface:

```text
SkillBuilder
```

textResponsibilities:

1. Analyze task
2. Define skill contract
3. Determine required permissions
4. Generate implementation
5. Generate tests
6. Run tests
7. Diagnose failures
8. Repair implementation
9. Re-run tests
10. Save verified skill

Do not put all of this logic into one massive Agent prompt.

Keep responsibilities separated where practical.

Suggested stages:

```text
TaskSpec
↓
SkillSpec
↓
SkillCode
↓
SkillTests
↓
Validation
↓
Repair
↓
VerifiedSkill
```

text---

# **6. Generated Skills**

Generated skills should initially prefer a constrained and predictable implementation format.

If OpenTeddy already uses Python skills, continue using Python.

A generated skill should ideally contain:

```text
skill metadata
implementation
input schema
output schema
permission declaration
tests
version
```

textExample logical structure:

```text
skills/generated/invoice_to_excel/
  skill.json
  main.py
  test_main.py
```

textAdapt to existing project conventions.

Do not introduce this exact folder structure if the repository already has a better established pattern.

---

# **7. Test Before Save**

A generated skill must NOT immediately become trusted or reusable.

Required flow:

```text
Generate
↓
Run test
↓
Pass?
```

textIf yes:

```text
mark verified
save/register skill
```

textIf no:

```text
capture error
↓
diagnose
↓
repair
↓
run test again
```

textSet a configurable maximum repair attempt count.

Suggested default:

```text
MAX_SKILL_REPAIR_ATTEMPTS = 3
```

textAvoid infinite loops.

If all attempts fail:

```text
do not register as verified
return structured failure
preserve useful diagnostics
```

text---

# **8. Self-Healing Existing Skills**

Existing generated skills may later break.

Example reasons:

```text
API changed
dependency changed
input changed
library updated
unexpected data
```

textWhen a previously verified generated skill fails:

```text
Execute
↓
Failure
↓
Capture Error
↓
Repairable?
↓
Repair Skill
↓
Test
↓
Create New Version
↓
Execute
```

textImportant:

Do not destructively overwrite the last known working version.

Support versioning such as:

```text
1.0.0
1.0.1
1.0.2
```

textKeep rollback possible.

For the first implementation, a simple version history is acceptable.

---

# **9. Priority 3 — Model Router**

Do not redesign OpenTeddy as Cloud-first.

Implement or extend model routing around:

**Task-first routing**

The router should support at minimum:

```text
LOCAL_PREFERRED
STRONG_MODEL_PREFERRED
LOCAL_ONLY
```

textSuggested conceptual decision:

### **LOCAL_PREFERRED**

Use for:

```text
simple classification
summarization
intent detection
skill lookup
simple planning
routine tasks
```

text### **STRONG_MODEL_PREFERRED**

Use for:

```text
new skill generation
skill repair
complex coding
complex reasoning
difficult planning
repeated local model failure
```

text### **LOCAL_ONLY**

Use when:

```text
user explicitly enabled Local Only
task/data policy prohibits cloud use
```

textIn Local Only mode:

No cloud model call is allowed, including hidden fallback behavior.

---

# **10. Cloud Intelligence → Local Capability**

This is a key architecture principle.

A strong cloud model may be used to teach Teddy a skill.

After the skill is learned:

```text
Future request
↓
Find existing skill
↓
Execute locally
↓
No need to regenerate
```

textDo not unnecessarily call Claude/GPT/Gemini again for every repeated task.

Example:

```text
First request:
"Convert these invoice PDFs into an Excel report."

→ no skill
→ strong model builds invoice_to_excel
→ validate
→ save

Second request:
"Do the same for these invoices."

→ find invoice_to_excel
→ execute locally
→ no Skill Builder
```

textThis behavior must be tested.

---

# **11. Priority 4 — Permission Model**

Each skill must declare what it needs.

Create a permission schema.

Suggested conceptual model:

```typescript
interface SkillPermissions {
  filesystem?: {
    read?: string[];
    write?: string[];
  };

  network?: {
    domains?: string[];
  };

  commands?: string[];

  credentials?: string[];

  services?: string[];
}
```

typescriptExample:

```json
{
  "filesystem": {
    "read": ["/workspace/input"],
    "write": ["/workspace/output"]
  },
  "network": {
    "domains": ["github.com"]
  },
  "credentials": ["github"]
}
```

jsonDo NOT grant unrestricted permissions by default.

Generated skills should request the minimum reasonable permissions.

For this branch, prioritize:

```text
permission schema
permission propagation
permission checks/interfaces
```

textA full polished permission UI is not required yet.

---

# **12. Runtime Abstraction**

Do not tightly couple generated skill execution to the current execution implementation.

Introduce or cleanly define a runtime abstraction.

Conceptually:

```typescript
interface SkillRuntime {
  execute(
    skill: SkillDefinition,
    input: unknown,
    context: RuntimeContext
  ): Promise<SkillExecutionResult>;
}
```

typescriptPotential implementations:

```text
NativeRuntime
DockerRuntime
OpenShellRuntime     // future
```

textFor this branch:

* Keep the current runtime working
* Introduce the abstraction if necessary
* Do NOT make NVIDIA OpenShell a hard dependency
* Do NOT spend significant time implementing OpenShell integration yet

The architecture should simply make future integration possible.

---

# **13. NVIDIA OpenShell**

Treat NVIDIA OpenShell as a future secure execution backend.

Do NOT:

```text
rewrite OpenTeddy around OpenShell
require OpenShell for normal execution
duplicate OpenShell's entire security architecture
```

textDo:

```text
make runtime replaceable
make permissions structured
keep execution separated from agent reasoning
```

textA future implementation should be able to add:

```text
OpenShellRuntime implements SkillRuntime
```

textwithout rewriting Skill Builder.

---

# **14. Observability**

Add useful structured logging/events around the learning lifecycle.

Suggested events:

```text
skill.search.started
skill.search.matched
skill.search.no_match

skill.build.started
skill.build.generated

skill.test.started
skill.test.failed
skill.test.passed

skill.repair.started
skill.repair.completed

skill.registered

skill.execution.started
skill.execution.failed
skill.execution.completed

model.route.selected
```

textDo not log credentials or sensitive user data.

---

# **15. Metrics**

Where reasonable, expose or record:

```text
Skill Generation Success Rate
Skill Execution Success Rate
Skill Repair Success Rate
Skill Reuse Rate
Average Repair Attempts
Model Used Per Skill
Cloud Cost Per Learned Skill
Time To Learn Skill
```

textDo not build a full analytics dashboard for this branch.

Structured internal metrics/events are enough.

---

# **16. MVP Acceptance Tests**

Add automated tests where practical.

At minimum, validate these scenarios.

## **Scenario A — Existing Skill**

Given:

```text
matching verified skill exists
```

textWhen:

```text
user submits compatible task
```

textThen:

```text
existing skill is selected
skill executes
Skill Builder is NOT called
```

text---

## **Scenario B — New Skill**

Given:

```text
no matching skill exists
```

textWhen:

```text
user submits task
```

textThen:

```text
Skill Builder runs
skill is generated
skill is tested
skill passes
skill is registered
skill executes
```

text---

## **Scenario C — Generated Skill Initially Fails**

Given:

```text
generated implementation has a test failure
```

textThen:

```text
error is captured
repair flow runs
new implementation is tested
verified version is saved
```

textMust enforce maximum repair attempts.

---

## **Scenario D — Skill Reuse**

Run the same logical task twice.

First execution:

```text
build skill
save skill
execute
```

textSecond execution:

```text
find existing skill
execute existing skill
```

textAssert:

```text
Skill Builder call count on second run = 0
```

textThis is one of the most important acceptance tests.

---

## **Scenario E — Local Only**

Given:

```text
Local Only = true
```

textThen:

```text
no cloud provider may be invoked
```

textTest this explicitly using provider mocks/spies.

---

## **Scenario F — Permission Declaration**

Generated skill must produce a structured permission declaration.

Ensure execution receives those permissions through runtime context.

---

## **Scenario G — Existing Skill Breaks**

Given:

```text
previous generated skill fails at runtime
```

textThen:

```text
repair flow can create a new version
old version remains recoverable
```

text---

# **17. Non-Goals**

Do NOT prioritize the following in this branch:

```text
WhatsApp integration
Discord integration
Slack integration
new chat channels
major UI redesign
large marketplace of skills
hundreds of built-in skills
adding many new LLM providers
building a custom sandbox platform
full NVIDIA OpenShell integration
rewriting the entire agent architecture
```

textKeep scope disciplined.

---

# **18. Engineering Principles**

Please follow these principles:

### **Preserve Existing Behavior**

Existing OpenTeddy workflows should continue working unless intentionally changed.

### **Minimal Refactor**

Refactor only where necessary to create clear abstractions.

### **Avoid God Objects**

Do not put model routing, skill generation, execution, testing, storage and permission logic into one class/service.

### **Strong Types**

Use existing type system and schemas wherever possible.

### **Dependency Injection**

LLM providers, registry, runtime, storage and test execution should be mockable.

### **Testability**

Design the Skill Builder so unit tests do not require real commercial LLM calls.

### **Configurable Policies**

Values such as:

```text
repair attempts
skill match threshold
model strategy
local-only mode
```

textshould not be hardcoded deep inside business logic.

---

# **19. Suggested Logical Architecture**

Use existing project naming where possible, but conceptually aim for:

```text
User Request
       │
       ▼
 Task Planner
       │
       ▼
 Skill Matcher
       │
   ┌───┴────┐
   │        │
Matched   No Match
   │        │
   │        ▼
   │   Skill Builder
   │        │
   │        ▼
   │   Model Router
   │        │
   │        ▼
   │   Generate Skill
   │        │
   │        ▼
   │      Tester
   │        │
   │     Fail ───→ Repair
   │        │          │
   │        └──────────┘
   │
   ▼
Skill Registry
   │
   ▼
Permission Check
   │
   ▼
Skill Runtime
   │
   ▼
Result Validator
   │
   ▼
User Result
```

text---

# **20. Deliverables**

Complete the branch with:

1. Repository architecture assessment
2. Implementation plan
3. Skill Registry changes
4. Skill Discovery / Matcher
5. Skill Builder
6. Generate → Test → Repair → Verify loop
7. Skill reuse
8. Model routing strategies
9. Local Only enforcement
10. Permission schema
11. Runtime abstraction
12. Skill versioning / repair foundation
13. Unit/integration tests
14. Updated documentation
15. Summary of files changed

---

# **21. Before Coding**

Before implementing major changes, first output:

```text
1. Current architecture discovered
2. Existing components that can be reused
3. Gaps versus this specification
4. Proposed minimal architecture
5. Files/modules expected to change
6. Implementation sequence
7. Risks / backwards-compatibility concerns
```

textThen proceed with implementation unless a genuinely blocking issue exists.

Do not ask for clarification for minor architecture choices.

Use the existing repository conventions and make the most reasonable engineering decision.

---

# **22. Completion Report**

At completion, provide:

```text
Branch:
<name>

Implemented:
- ...

Architecture changes:
- ...

Tests added:
- ...

Tests passing:
- ...

Known limitations:
- ...

Future work:
- OpenShell runtime
- richer permission UI
- stronger semantic skill matching
- skill marketplace / sharing if desired
```

textAlso include the exact commands needed to run relevant tests locally.

---

# **Final Product Requirement**

The most important behavioral requirement is:

**OpenTeddy must stop thinking of skills as static plugins and start treating skills as capabilities Teddy can learn.**

The desired loop is:

```text
I know how
→ do it

I don't know how
→ learn it
→ test it
→ remember it
→ do it

I learned it before
→ reuse it
```

textThe North Star is:

**Task → Learn → Verify → Execute → Reuse**

Do not optimize for the number of models or number of integrations.

Optimize for:

**How reliably can OpenTeddy turn a new user request into a reusable capability?**

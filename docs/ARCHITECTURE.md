# CVnator — Architecture

## The one-sentence version

The AI decides, deterministic tools execute, the system validates every step, and
failures produce recovery or an honest escalation — never a silent success.

## Why this is not a macro

A form-filling macro is a decision table: *if element X exists, click X*. It works
until the page changes, and then it fails silently or does the wrong thing.

CVnator inverts that. The browser layer produces **evidence** — a labelled inventory
of what is on the page — and a reasoning model decides what that evidence means and
what to do about it. Playwright, the DOM, the accessibility tree and the vision model
are instruments the agent reads; none of them decides anything.

The concrete consequence: when the mock ATS silently refused a form submission
because an HTML5 `step` constraint rejected `6.2`, the fix was not a new selector. It
was making the constraint *visible* — surfacing `el.validationMessage` into the
inventory — after which the model diagnosed and corrected it on its own.

## Layers

```
apps            frontend (React/Vite)  ·  FastAPI + SSE
core/agent      loop · policy engine · memory · watchdog · validators · orchestrator
browser         controller (Playwright) · perception · tools · vision fallback
ai              provider abstraction · gateway · resource manager (VRAM)
resume          extract · parser · ats · optimizer · fact validator · entailment · renderer
jobs            matching · dedupe · prioritization
platforms       base/registry · mock · generic · linkedin* · gupy*      (* scaffolds)
storage         artifacts (GC) · credentials (OS vault)
db              models · migrations · repositories
mockats         local test ATS
```

## The agent loop

```
OBSERVE → UNDERSTAND → PLAN → ACT → VALIDATE → UPDATE MEMORY → CONTINUE
```

Each cycle carries `current_goal`, `current_state`, `page_state`, `available_actions`,
`last_action`, `expected_result`, `observed_result`, `retry_count`, and is recorded
for audit.

**PLAN** emits exactly one schema-validated `AgentDecision`. **ACT** runs it only
after the policy engine rules on it. **VALIDATE** reads the tool's own report of what
actually changed — never the agent's expectation.

## Five decisions worth explaining

### 1. Identity data is structurally unwritable by the model

The resume optimizer receives the profile with every bullet numbered, and may only
answer with *references*: reorder these experiences, rewrite bullet 1 of experience 2
as this text, put SQL before Python. Company names, job titles, dates, degrees and
certifications are then copied verbatim from the master profile by `build_document`.

An invented employer is therefore **impossible by construction**, not merely
discouraged by a prompt. What remains for the fact validator to police is narrow: the
headline, the summary, and the rewritten bullets.

### 2. The fact validator is three layers, and only the model-backed one can be wrong

1. **Structural** — the invariant above.
2. **Lexical/numeric** (deterministic) — technology claims the profile cannot support;
   duration claims larger than the profile justifies; work-authorization assertions
   when none is on file.
3. **Entailment** (model-backed) — used *only* where lexical comparison is meaningless:
   a bullet translated into the posting's language shares almost no tokens with its
   source, and a coverage check would read that legitimate translation as fabrication.

Layer 3 can only **add** violations, never clear one. A model is not permitted to
authorise a claim. If the entailment call fails, the rewrite is treated as unverified
and blocked — it fails closed.

The measure for rewrites is *coverage*, not similarity: expanding a bullet is
legitimate, replacing it is not. The specification's own example rewrite passes;
"Led a team of 12 engineers" does not.

### 3. The policy engine has no LLM in it

A prompt can be talked around; a conditional cannot. The engine runs before every
action and enforces, in order: prohibitions (never interact with a CAPTCHA or MFA
challenge), factual integrity (the value about to be typed is checked against the
profile), authorization (account creation, legal consent, final submission), and
pacing.

When it blocks a value it tells the agent **not to rephrase it** — otherwise the model
simply tries a softer wording of the same fabrication.

### 4. Nothing is assumed to have worked

Every tool re-observes the page and reports what genuinely changed. A click that
Playwright performed without raising is still a failure if the page did not move —
and when it did not, the tool names the fields the browser rejected rather than
leaving the agent to retry blindly.

An application reaches `SUBMITTED` only on real evidence: a confirmation phrase, a
reference number, a confirmation route, or the form disappearing. Absent that, the
user is told it could not be confirmed.

### 5. Vision is a fallback, not a habit

Perception is layered DOM → accessibility tree → browser state → screenshot → VLM, and
the expensive layers are reached only when the cheap ones genuinely fail. This is not
a micro-optimisation: on a 12 GB card, loading the VLM can evict the reasoning model.

An early build treated "the page fingerprint did not change" as a stall, which fired
on every well-behaved form — filling a text input does not change page structure — and
dragged the vision model in six times per application. Progress is now defined as *the
tool reported success*.

## Running models on 12 GB

Models are addressed by **role** (reasoning, vision, resume, translation), never by
name. The `ResourceManager` tracks residency against a VRAM budget, evicts
least-recently-used models to make room, and drops idle ones — using Ollama's
`keep_alive` for real eviction and `/api/ps` for real usage figures rather than
estimates.

Two things a local reasoning model taught us the hard way, both now defaults in
`AIGateway.structured`:

* **`think=False`.** Reasoning models route chain-of-thought into a separate channel
  and will happily spend the entire token budget there, returning empty content. For a
  schema-constrained call the schema *is* the reasoning scaffold.
* **Strictified schemas.** Pydantic omits defaulted fields from `required`, and a
  grammar-constrained decoder reads that as permission to stop early — answering `{}`
  and being formally correct. Every field is marked required for the model; the Python
  side keeps its defaults.

## Data model

Rich, evolving structures (the master profile, a match report, an ATS analysis) are
JSON columns; anything queried or joined on is a real column.

`ResumeVersion` stores the **recipe** for a tailored CV — source profile version, job
description hash, transformation summary, ATS scores before and after, template, file
hash — not the PDF. That is what makes deleting the artifact safe: the file is
disposable, the recipe is permanent, and the CV can be rebuilt byte-for-byte.

## What is deliberately absent

* **No anti-bot evasion.** No stealth flags, no fingerprint spoofing, no CAPTCHA
  solving. If a platform declines to be automated, the agent stops and tells the user.
* **No credential exposure.** Passwords live in the OS credential vault. The agent
  receives an opaque handle; the browser layer resolves it at the moment of typing. A
  redaction filter on the root log handlers covers third-party libraries and tracebacks.
* **No claimed knowledge of ATS algorithms.** The ATS score is CVnator's own coverage
  estimate, useful for comparing two versions of a resume against one posting. It
  carries that disclaimer in its own serialised output.

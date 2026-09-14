# Decision log

Architectural decisions and the reasoning behind them. Several were forced by things
that only showed up once the agent was running against a real browser and a real
local model; those are marked **[from a run]** and are the most useful entries here.

---

## ADR-001 — Python for the agent, React for the UI

**Decision.** One Python process runs the API, the agent and the browser; the UI is a
separate React build served as static files from that process.

**Why.** Playwright, the document toolchain (pypdf, python-docx, reportlab) and the
local-model ecosystem all have their centre of gravity in Python. The agent needs to
hold a browser and an event loop open for hours, which is a poor fit for a request
model. One process means one thing to start.

**Cost.** A Python GIL-bound process doing browser work and API serving together. In
practice the agent is I/O-bound on the model and the browser, and CPU-bound work
(rendering) is pushed to a thread.

---

## ADR-002 — Models are addressed by role, never by name

**Decision.** `ModelRole.REASONING / VISION / RESUME / TRANSLATION`, resolved by a
gateway. Providers implement one interface.

**Why.** The right model per role changes constantly, and a 12 GB card forces
different trade-offs from a 24 GB one. Swapping Ollama for llama.cpp, vLLM or a
remote endpoint should be one class, not a refactor.

---

## ADR-003 — SQLite through SQLAlchemy, with JSON for evolving shapes

**Decision.** SQLite for the MVP; every access through the ORM; Alembic migrations
from the start. Rich structures (profile, match report, ATS analysis) are JSON
columns, anything queried or joined on is a real column.

**Why.** Local-first means a file, not a server. The ORM keeps the door open. The
JSON split avoids a migration every time a model gains a field, without giving up
indexing on the things actually queried.

---

## ADR-004 — The model cannot write identity data

**Decision.** The resume optimizer receives numbered bullets and answers only with
references (reorder, rewrite bullet *n*, order skills). Company names, titles, dates,
degrees and certifications are copied from the profile by `build_document`.

**Why.** A prompt saying "do not invent employers" is a request. Not giving the model
a field to write the employer into is a guarantee. This shrinks the fact validator's
job to the prose the model actually produced.

**Cost.** The optimizer cannot restructure a CV freely — only reorder and rephrase.
That is the intended limit.

---

## ADR-005 — Rewrites are judged by coverage, not similarity **[from a run]**

**Decision.** A rewrite must retain ≥60% of the original's *content words*.

**Why.** The first implementation used `fuzz.token_set_ratio`, which is symmetric and
therefore penalises expansion. The specification's own example of an acceptable
rewrite scored 54 against a floor of 55 — a false positive on the exact case the
product is supposed to allow. Coverage is asymmetric: expanding is free, replacing is
not.

---

## ADR-006 — Entailment is a third layer, and it can only add violations **[from a run]**

**Decision.** When a rewrite is in a different language from its source, skip lexical
coverage and ask the model whether the rewrite asserts anything the original does
not. A failed or unavailable check blocks the rewrite.

**Why.** The candidate's CV is in Portuguese; postings are often in English.
Translating a truthful bullet is required behaviour, and shares almost no tokens with
its source — the deterministic layer read every legitimate translation as a
fabrication.

**The constraint that makes this safe.** Layer 3 can only add violations. A model is
never allowed to clear one raised deterministically, and never to authorise a claim.
Number preservation and the "no new technology" rule stay in force across languages,
because digits and technology names do not translate.

---

## ADR-007 — The policy engine contains no LLM

**Decision.** A plain-Python layer rules on every proposed action before it runs.

**Why.** The failure mode of a prompt-based guardrail is that the model argues its way
around it, and the failure is invisible. A conditional cannot be persuaded. It also
makes every refusal explainable: a rule name and a reason, not a vibe.

**Detail that mattered.** When the engine blocks a value it tells the agent *not to
rephrase it*. Without that, the model reliably tried a softer wording of the same
fabrication.

---

## ADR-008 — Structured output needs strict schemas and no thinking **[from a run]**

**Decision.** `AIGateway.structured` sets `think=False` and marks every schema field
required before handing the schema to the model.

**Why.** Two separate failures, both silent:

* `qwen3.5:9b` is a reasoning model. It routed ~5,800 tokens into its `thinking`
  channel and returned **empty** `content`, surfacing as "model returned an empty
  response". For a schema-constrained call the schema is the reasoning scaffold.
* Pydantic omits defaulted fields from `required`. A grammar-constrained decoder reads
  that as permission to stop, and the model answered `{}` — formally valid, entirely
  useless. Forcing every field required fixed extraction quality outright: the ATS
  analyser went from zero requirements to correctly classified required/preferred
  ones, and a poorly-matched job that had scored 58% correctly dropped to 31%.

The provider now also raises a specific error when content is empty but thinking is
not, so this failure names itself next time.

---

## ADR-009 — Progress means "the tool reported success" **[from a run]**

**Decision.** The watchdog's stall detector requires consecutive *failed* actions on
an unchanged page. Loop detection keys on (page, action, target).

**Why.** The first version treated an unchanged page fingerprint as a stall. Filling a
text input does not change page structure, so it fired on every well-behaved form and
pulled the vision model in six times per application — evicting the reasoning model
from VRAM each time. The expensive fallback was being triggered by correct behaviour.

---

## ADR-010 — Generated selectors must be provably unique **[from a run]**

**Decision.** The perception layer tests each candidate selector with
`querySelectorAll` and falls back to a full structural path when it is not unique.

**Why.** The mock ATS has two submit buttons both named `action` (Back and Continue).
The original generator emitted `button[name="action"]` for both, and `.first` resolved
to Back — so the agent kept "advancing" backwards while the tool reported success.
Real forms do this constantly with radio groups and wizard navigation.

---

## ADR-011 — HTML5 constraint validation must be surfaced **[from a run]**

**Decision.** The element inventory carries `invalid`, `validationMessage` and the
`min/max/step/pattern/maxlength` constraints, and native validation failures are
promoted into the page's alert list.

**Why.** The mock's "years of experience" input has `step="0.5"`. The agent entered
the truthful `6.2`; the browser silently refused to submit and showed a native bubble
that never enters the DOM. From the agent's side a click simply did nothing, with no
reason available anywhere. With the constraint visible, the model diagnosed it
unaided: *"invalid per the page's step constraint (0.5)"*.

**Follow-on.** This created a genuine product question — truthfulness versus a form
constraint. The rule is to enter the nearest permitted value that does **not**
overstate: round down, never up. 6.2 becomes 6.0, not 6.5.

---

## ADR-012 — Vision is a fallback, and the agent is told what it already has **[from a run]**

**Decision.** Screenshots go to the VLM only when the cheap layers genuinely fail. The
agent's context states when a tailored resume has already been generated.

**Why (second half).** The agent reached the upload step, found a file input, tried to
type into it, and then escalated to the user asking for a file path — while the
orchestrator was already holding the generated PDF. It had no way to know the file
existed. Autonomy is lost as easily to missing context as to a missing capability.

---

## ADR-013 — Submission requires evidence

**Decision.** `SUBMITTED` is reachable only on a confirmation phrase, a reference
number, a confirmation route, or the form disappearing. Otherwise the user is told it
could not be confirmed.

**Why.** "The click did not raise" is not evidence. A history that records unverified
submissions is worse than no history, because the user stops checking.

---

## ADR-014 — Store the recipe, not the artifact

**Decision.** `ResumeVersion` records source profile version, job description hash,
transformation summary, ATS scores, template and file hash. The PDF is disposable.

**Why.** It reconciles "delete tailored CVs after submission" with "be able to explain
and reproduce what was sent". The file can be rebuilt byte-for-byte.

---

## ADR-015 — Real-platform adapters ship disabled

**Decision.** LinkedIn, Indeed, Glassdoor and Gupy adapters exist, are documented as
unvalidated scaffolds, and are not registered on import.

**Why.** Those platforms' terms prohibit automated access, and their markup changes
faster than any committed selector. Registering them by default would imply a level of
support that has not been earned, and would make a consequential decision on the
user's behalf. The mock adapter and the generic AI-driven fallback are the paths that
are actually ready.

**What is deliberately absent.** No stealth flags, no fingerprint spoofing, no CAPTCHA
solving. A block becomes an escalation, not a workaround.

---

## ADR-016 — Corrections and settled values must survive the turn **[from a run]**

**Decision.** The per-step context carries two additional blocks: the values already
entered in this application, and any value the policy engine has insisted on.
`PageState.form_progress()` also renders an explicit checklist of which required
fields remain.

**Why.** The first full end-to-end run did not finish. The pipeline worked —
discovery, matching, a fact-checked tailored CV, four form steps, a verified upload —
and the policy engine correctly blocked nine attempts to overstate experience. But
the agent then exhausted its 60-step budget and escalated, because each turn
re-derived the blocked value from the profile and tried it again. The context said
what the page looked like, not what had already been settled.

The escalation itself was correct behaviour: the run ended in `WAITING_USER` with an
honest reason, not a fabricated success. But 21 tool retries across 93 steps is a
system spending its budget arguing with itself.

**The general lesson.** Autonomy is lost to missing context at least as often as to
missing capability. Both of the substantial autonomy failures observed — this one and
the agent asking for a resume path it already had (ADR-012) — were cases where the
runtime knew something and had not told the agent.

**Still open.** Whether these changes are sufficient is an empirical question, and
the honest answer is that it needs more runs than one. The step budget is deliberately
left at 60: a larger budget would hide oscillation rather than fix it.

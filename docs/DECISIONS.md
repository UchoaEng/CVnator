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

---

## ADR-017 — Identity fields are checked positively, not negatively **[from a run]**

**Decision.** When a field label asks for the candidate's own employer or job title,
the value must *match something in the profile*. Anything else is blocked, and the
message names the value that should have been used.

**Why.** Asked for "Company" under *Most recent experience*, the agent filled in
**the company it was applying to**. The mock ATS received
`current_company = "Lumina Tecnologia"` where the candidate's employer is
"Northbridge Logística".

Nothing caught it, and the reasons are worth stating plainly:

* It contains no unsupported technology, so the skill check passed.
* It contains no number, so the duration check passed.
* The entity check only fires on attribution phrasing ("at <Company>") and is
  warn-level anyway.
* The value is a real company name in plausible form. **There is no negative signal
  to detect.**

This is the shape of error a keyword-based guard cannot see: a well-formed, confident,
entirely wrong answer. The only defence is to require a positive match against the
profile rather than to scan for something suspicious. Fuzzy matching handles accents
and spacing so that "Northbridge Logistica" is not blocked for missing an í.

**The general principle.** Where a field has a knowable correct answer, validate
against that answer. Reserve pattern-matching for free text, where there is nothing to
match against.

---

## ADR-018 — Missing context looks exactly like missing capability **[from a run]**

**Decision.** The profile digest renders skills with their years — `SQL (5y)` — and
says explicitly to prefer the per-skill number over the total.

**Why.** The agent escalated to the user with: *"the profile facts list total
experience (6.2) but do not specify how many years are specifically with SQL. I cannot
guess a truthful value."*

That is textbook-correct behaviour — refusing to guess and asking instead. The problem
is that the answer was already on file. The digest listed skill *names* only, so the
agent could not see `SQL: 5.0`. The system had the fact and did not show it.

Together with ADR-012 (asking for a resume path the runtime already held) and ADR-016
(re-deriving a value the policy engine had already corrected), this is now the third
autonomy failure of the same shape. Every one of them was the runtime knowing
something and not telling the agent — not the agent being incapable.

When an escalation looks unnecessary, the first question is not "why did the model get
this wrong" but "what did we fail to put in front of it".

---

## ADR-019 — Only a signal that can mean nothing else confirms a submission **[from a run]**

**Decision.** `validate_after_submit` confirms on an explicit confirmation phrase or
a reference number, and on nothing else. Weak signals are still recorded as evidence,
but no combination of them confirms.

**Why.** The original rule was "a phrase or a reference, **or any two weaker
signals**". That produced the worst failure observed in this project: the agent
advanced from the screening-questions step to the review step, and the validator saw

* the URL changed to `.../apply/<token>/5`
* the form's inputs were gone

— two weak signals agreeing — and recorded an application as **SUBMITTED that had
submitted nothing**. The mock ATS still held `submitted: false`.

Both signals are real. Both are also exactly what a review page looks like. Their
conjunction carries no more meaning than either alone.

**What caught it and what did not.** The state machine refused the illegal
`APPLYING → SUBMITTED` edge, so the database was never corrupted. But
`_transition` returned `None`, and the caller went on to act on the agent's *claim*:
it deleted the tailored CV for an application that was never sent. `_transition` now
returns whether the edge was recorded, and everything downstream of "submitted" is
gated on that.

**The principle.** Prefer a missed confirmation to a false one. Failing to notice a
real submission is a nuisance the user can correct; recording one that never happened
is a lie the user has no way to detect, and it poisons the history they rely on.

---

## ADR-020 — A refusal the agent cannot act on still costs the application **[from a run]**

**Decision.** A blocked value's remedy depends on the shape of the field. A blocked
paragraph is told which term to remove, to write the answer again without it, and not
to substitute a synonym. Short values keep the terse "use a supported value or ask the
user".

**Why.** The fact validator correctly refused an ETL claim the candidate cannot
support — four times. Each time it said *"Do not rephrase it — either use a supported
value or request the answer from the user"*, which is sound advice for a number and
meaningless for a motivation paragraph: there is no "supported value" to substitute,
and the user has nothing to supply. The agent resubmitted prose containing the same
term until the retry budget ran out and the application failed.

With the shape-aware remedy, the next run hit the same block once and recovered in a
single turn: *"Rewriting the answer for [e10] without 'etl'"*.

**The principle.** A guard is not finished when it refuses correctly. Refusing
correctly and leaving the caller with no path forward converts a small factual error
into a lost application. The remedy is part of the guard.

## ADR-021 — Repeat the run before believing it **[from a run]**

**Decision.** Reliability is measured by `scripts/measure_reliability.py`: N cold-start
runs, each a separate subprocess with its own data directory, database, browser profile
and artifact store, aggregated into a success rate and the agent-evaluation rates of
spec section 74. A run with zero applications is skipped rather than scored as a
perfect 0.0; a crashed run stays a failure and still divides the denominator.

**Why.** One successful run proved the path existed and was reported as such. Repeating
it exposed three defects the single run had hidden — the first repeat run lost both of
its applications, and a later batch, with two of the three fixed, still lost one
application in ten:

1. **The session limit counted attempts, not submissions.** `applications_this_session`
   was fed from `applications_started`, so the Nth application was blocked from
   submitting by its own start. A limit of 2 delivered one application and abandoned
   the second at the review screen. The earlier run's second application sitting in
   `APPLYING` was read as correct pacing; it was this bug.
2. **An unconfirmed submit offered giving up as an option.** The agent clicked *next* on
   the last form step, landed on the review page, and was told only that the submission
   *"could not be confirmed ... or report failure"*. It reported failure, one control
   short of the end.
3. **One malformed answer ended the application.** The model narrated past the token
   budget mid-JSON; the parse failed; the application was recorded `FAILED` with no
   retry.

After the fixes: **5 of 5 runs, 10 of 10 applications submitted**, no human
intervention, 23–24 steps per application.

**The principle.** Defects two and three are ADR-020 again, in new places: a correct
refusal that leaves no move. That makes three instances, which is enough to state it as
a rule rather than an anecdote — **every refusal path must name the next action**, and a
guard's remedy deserves the same test coverage as its trigger.

The narrower lesson is about evidence. A single green run measures that success is
*possible*, and it is easy to mistake for a measure of how often it *happens*. The two
differ by whatever the run happened not to hit — here, three defects and a false
reading of a fourth behaviour. Nothing may be called reliable on one observation.

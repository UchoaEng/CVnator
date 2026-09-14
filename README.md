# CVnator

An autonomous, computer-using job application agent. It searches for jobs, reads
them, scores them against your real experience, tailors a CV that stays truthful,
fills in multi-step application forms in a real browser, validates before and after
submitting, and asks you only when it genuinely cannot proceed.

Everything runs on your machine: local models via Ollama, a local database, local
files. Nothing leaves the computer unless you explicitly configure a remote endpoint,
and the code says so out loud when you do.

## What works today

The pipeline runs against a local mock ATS with a real local model and a real
Chromium:

```
CV → Candidate Master Profile → job discovery → job intelligence → match scoring
→ ATS analysis → tailored resume → fact validation → PDF/DOCX render
→ browser agent → multi-step form → upload → pre-submission validation
→ submit → post-submission validation → history → artifact cleanup
```

**Verified end to end in a single autonomous run** against the mock ATS, with a local
model and a real Chromium:

```
3 jobs discovered  →  scored 71% / 31% (rejected) / 100%
                   →  tailored CV, fact-checked, ATS 81.3% → 81.9%
                   →  5-step form, resume uploaded
                   →  1 fabricated claim blocked, recovered in one turn
                   →  pre-submission validation → submitted
                   →  confirmed on real evidence (reference MA-951122)
                   →  APPLYING → SUBMITTING → SUBMITTED
                   →  temp files deleted, recipe retained
```

The in-flight second application kept its resume, as it should — only a confirmed
submission triggers cleanup.

One run is not a reliability claim. Expect to hit cases this has not seen; the
failure modes are designed to be visible (an honest escalation, a recorded failure)
rather than silent.

**Scaffolds, clearly marked:** the LinkedIn, Indeed, Glassdoor and Gupy adapters.
Written from the public structure of those sites, **not** validated against
production, and not registered by default — enabling them is an explicit act. See the
note on terms of service below.

188 tests (unit + browser integration against the mock ATS).

## Requirements

- Python 3.12–3.14, Node 20+
- [Ollama](https://ollama.com) running locally
- ~8 GB of VRAM for the default models (the resource manager works within a budget you set)

```bash
ollama pull qwen3.5:9b     # reasoning, resume, translation
ollama pull gemma3:4b      # vision fallback
```

## Setup

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
.venv/Scripts/python -m playwright install chromium
cp .env.example .env
.venv/Scripts/python -m alembic upgrade head
cd frontend && npm install && npm run build && cd ..
```

## Run it

```bash
.venv/Scripts/python -m cvnator.cli serve
```

Then open http://127.0.0.1:8088 and: upload your CV → review the extracted profile →
create a job search → press **Start**.

For UI development, run the API and `npm run dev` in `frontend/` side by side; Vite
proxies `/api` to the Python process.

## Try it safely, against the mock ATS

```bash
.venv/Scripts/python -m cvnator.mockats        # terminal 1
.venv/Scripts/python scripts/run_vertical_slice.py   # terminal 2
```

This exercises the full pipeline without touching anyone's real service.

## The rules this thing follows

These are enforced in code — in the fact validator and the policy engine — not just
asked for in a prompt.

**It will not invent anything about you.** Not a job, a technology, a certification, a
language, a salary, or a number of years. It can rephrase what is true — "created
operational dashboards" may become "developed and maintained dashboards for
operational KPI analysis" — and it refuses anything that adds a fact. Company names,
job titles and dates are copied from your profile; the model is never given a way to
write them.

**It will not defeat a CAPTCHA, an MFA prompt, or a bot check.** It pauses and asks
you. There are no stealth flags and no fingerprint spoofing. If a platform says it is
blocking automation, the agent stops and tells you.

**It will not claim an action succeeded because it ran it.** A click that changed
nothing is a failure. An application is marked submitted only on real evidence — a
confirmation message, a reference number, a confirmation page.

**It will not hide failures.** Failed and abandoned applications are recorded with the
reason.

## Dry run

Dry run is **on by default**. The agent searches, matches, generates CVs and fills
forms, but the final submit is blocked by the policy engine. Turn it off in Settings
when you have seen enough to trust it.

## ATS scores are estimates

No ATS vendor publishes its ranking algorithm, and CVnator does not pretend to
reproduce one. The score is an internal coverage estimate — useful for comparing two
versions of your CV against the same posting, and nothing more. It is not a
probability of getting an interview.

You will also see modest improvements (81% → 82%) more often than dramatic ones. That
is the honesty constraint doing its job: the gap is usually made of skills you do not
have, and the system will not add them.

## A note on real platforms

LinkedIn, Indeed and Glassdoor prohibit automated access in their terms of service.
Using the scaffolded adapters against a real account may get that account restricted.
That is a decision only you can make, which is why those adapters ship disabled and
why there is no code here to help you avoid detection. The mock ATS and the generic
adapter — for career pages and ATS products that do not forbid it — are the paths that
are ready to use.

## Your data

- Your master CV and profile stay in `data/` on your machine.
- Credentials go to the OS credential vault (Windows Credential Manager, macOS
  Keychain, Secret Service) — never to the database, the logs, or a model prompt.
- Logs are filtered for anything that looks like a secret before being written.
- Tailored CVs are temporary by default and deleted once submission is confirmed; the
  recipe to rebuild them is kept.

## Development

```bash
.venv/Scripts/python -m pytest          # tests
.venv/Scripts/python -m ruff check src/ # lint
.venv/Scripts/python -m mypy            # types
cd frontend && npm run typecheck
```

`docs/ARCHITECTURE.md` explains the design decisions and why they are what they are.

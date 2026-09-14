# CVnator — working notes

Autonomous job application agent. Local-first: local models (Ollama), local SQLite,
local files.

## Commands

```bash
.venv/Scripts/python -m cvnator.cli serve       # API + UI on :8088
.venv/Scripts/python -m cvnator.mockats         # test ATS on :8099
.venv/Scripts/python -m pytest                  # tests
.venv/Scripts/python -m ruff check src/ tests/  # lint
.venv/Scripts/python -m mypy                    # types
.venv/Scripts/python -m alembic upgrade head    # migrations
cd frontend && npm run build                    # UI
python scripts/run_vertical_slice.py            # full pipeline vs mock ATS
```

## Invariants — do not weaken these

1. **The model cannot write identity data.** The resume optimizer answers with
   references (reorder, rewrite bullet *n*); company names, titles and dates are
   copied from the profile in `build_document`. Never add a field that lets the model
   emit an employer name.
2. **Entailment can only add violations.** `cvnator.resume.entailment` is consulted
   where lexical checks cannot apply (translations). It must never clear a violation
   raised deterministically, and it fails closed.
3. **The policy engine contains no LLM.** `cvnator.agent.policy` is plain Python and
   runs before every action.
4. **Nothing is assumed to have worked.** Tools re-observe and report what actually
   changed. `SUBMITTED` requires evidence from the page.
5. **No anti-bot evasion, ever.** No stealth flags, no CAPTCHA solving. A block
   becomes an escalation.
6. **Secrets never reach a prompt or a log.** Credentials live in the OS vault; the
   agent gets an opaque handle.

## Gotchas learned the hard way

- `AIGateway.structured` sets `think=False` and strictifies the schema. Removing
  either breaks local models silently (empty content, or `{}` responses).
- Playwright ≥1.52 has no `page.accessibility`; the a11y tree comes from CDP.
- Generated selectors are verified unique in JS. Forms reuse `name` across buttons
  and radio groups.
- Filling a text input does not change the page fingerprint. Progress = the tool
  reported success (see `Watchdog`).
- HTML5 constraint violations never enter the DOM. They are read via
  `checkValidity()`/`validationMessage` and promoted into `PageState.alerts`.
- Required-ness comes from the attribute *or* a `*` in the label (`required_source`).

## Layout

`src/cvnator/{domain,ai,browser,agent,jobs,resume,platforms,storage,db,api,mockats}`,
`frontend/` (React+Vite), `tests/{unit,integration}`, `docs/`.

Real-platform adapters (linkedin, indeed, glassdoor, gupy) are unvalidated scaffolds
and are deliberately **not registered on import**. See `docs/DECISIONS.md` ADR-015.

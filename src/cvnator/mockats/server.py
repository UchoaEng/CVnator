"""A local mock Applicant Tracking System for developing and testing the agent.

It deliberately reproduces the things that make real application forms hard:

* a multi-step wizard with server-side validation and a real error state
* a select, a radio group, checkboxes and a required consent box
* a file upload that rejects anything but PDF/DOCX
* one input whose ``<label>`` is not associated via ``for``, so the agent has to
  infer the label from layout rather than from the accessibility tree alone
* a free-text question that cannot be answered from a lookup table
* a confirmation screen with a reference number

Run it with ``python -m cvnator.mockats``. It is a test fixture, not a product
feature, and it never leaves the machine.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import DictLoader, Environment

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = (".pdf", ".docx")


@dataclass
class MockJob:
    job_id: str
    title: str
    company: str
    location: str
    remote: str
    description: str
    seniority: str = "mid"
    language: str = "en"
    posted_at: str = "2026-09-01"


MOCK_JOBS: dict[str, MockJob] = {
    "MJ-1001": MockJob(
        job_id="MJ-1001",
        title="Data Analyst",
        company="Northwind Analytics",
        location="São Paulo, BR",
        remote="remote",
        description=(
            "We are hiring a Data Analyst to own operational reporting.\n\n"
            "Responsibilities:\n"
            "- Build and maintain dashboards for operational KPIs\n"
            "- Model data in SQL and validate data quality\n"
            "- Partner with engineering on observability of data pipelines\n"
            "- Present findings to non-technical stakeholders\n\n"
            "Requirements (required):\n"
            "- 2+ years of experience with SQL\n"
            "- Hands-on experience with Python for data analysis\n"
            "- Experience building dashboards (Grafana, Power BI or similar)\n"
            "- Professional working proficiency in English\n\n"
            "Preferred:\n"
            "- Experience with Prometheus or other observability tooling\n"
            "- Familiarity with dbt or Airflow\n"
            "- AWS exposure\n"
        ),
    ),
    "MJ-1002": MockJob(
        job_id="MJ-1002",
        title="Observability Engineer",
        company="Cobalt Systems",
        location="Remote, LATAM",
        remote="remote",
        description=(
            "Observability Engineer to build the telemetry platform.\n\n"
            "Requirements (required):\n"
            "- Strong experience with Prometheus and Grafana\n"
            "- Kubernetes in production\n"
            "- Go or Python\n"
            "- 5+ years in SRE or platform engineering\n\n"
            "Preferred:\n"
            "- OpenTelemetry\n"
            "- Terraform\n"
        ),
    ),
    "MJ-1003": MockJob(
        job_id="MJ-1003",
        title="Analista de Dados Pleno",
        company="Lumina Tecnologia",
        location="Campinas, SP",
        remote="hybrid",
        language="pt",
        description=(
            "Buscamos Analista de Dados Pleno para atuar com indicadores operacionais.\n\n"
            "Responsabilidades:\n"
            "- Criar e manter dashboards de indicadores operacionais\n"
            "- Modelagem de dados em SQL\n"
            "- Automatizar rotinas de extração em Python\n\n"
            "Requisitos obrigatórios:\n"
            "- Experiência com SQL\n"
            "- Experiência com Python\n"
            "- Experiência com ferramentas de visualização de dados\n\n"
            "Desejáveis:\n"
            "- Conhecimento em Grafana\n"
            "- Inglês intermediário\n"
        ),
    ),
}


@dataclass
class MockApplication:
    token: str
    job_id: str
    step: int = 1
    data: dict[str, Any] = field(default_factory=dict)
    resume_filename: str = ""
    resume_size: int = 0
    submitted: bool = False
    reference: str = ""


APPLICATIONS: dict[str, MockApplication] = {}

_BASE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} — MockATS</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1a1d21; }
  .wrap { max-width: 720px; margin: 0 auto; padding: 24px 16px 64px; }
  header { background: #12243b; color: #fff; padding: 16px; }
  header .brand { font-weight: 700; letter-spacing: .04em; }
  .card { background: #fff; border: 1px solid #dfe3e8; border-radius: 10px; padding: 20px; margin-bottom: 16px; }
  label { display:block; font-weight:600; margin: 14px 0 4px; font-size: 14px; }
  input[type=text], input[type=email], input[type=tel], input[type=number], select, textarea {
    width: 100%; padding: 9px 10px; border:1px solid #c6ccd4; border-radius:6px; font-size:14px;
    box-sizing: border-box; font-family: inherit;
  }
  textarea { min-height: 96px; }
  button { background:#1a6dff; color:#fff; border:0; padding:10px 18px; border-radius:6px;
           font-size:14px; cursor:pointer; font-weight:600; }
  button.secondary { background:#6b7280; }
  .err { background:#fdeaea; border:1px solid #f3b7b7; color:#8a1c1c; padding:10px 12px;
         border-radius:6px; margin-bottom:12px; }
  .steps { display:flex; gap:8px; font-size:12px; color:#6b7280; margin-bottom:14px; flex-wrap:wrap; }
  .steps b { color:#1a6dff; }
  .muted { color:#6b7280; font-size:13px; }
  .ok { background:#e8f6ed; border:1px solid #a9ddbd; color:#1c6b3c; padding:14px; border-radius:8px; }
  pre { white-space: pre-wrap; font-family: inherit; }
  fieldset { border:1px solid #dfe3e8; border-radius:8px; margin: 14px 0; }
  .row { display:flex; gap:12px; flex-wrap: wrap; }
  .row > div { flex: 1 1 200px; }
</style>
</head>
<body>
<header><div class="wrap" style="padding:0 16px"><span class="brand">MOCKATS</span>
  <span class="muted" style="color:#9fb3cd"> · local test harness</span></div></header>
<div class="wrap">{% block content %}{% endblock %}</div>
</body>
</html>
"""

_JOB_LIST = """
{% extends "base" %}{% block content %}
<div class="card">
  <h1>Open positions</h1>
  <p class="muted">Local mock listings for agent testing.</p>
  <ul>
  {% for job in jobs %}
    <li style="margin:10px 0">
      <a href="/jobs/{{ job.job_id }}" id="job-link-{{ job.job_id }}">{{ job.title }}</a>
      — {{ job.company }} · {{ job.location }} · {{ job.remote }}
    </li>
  {% endfor %}
  </ul>
</div>
{% endblock %}
"""

_JOB_DETAIL = """
{% extends "base" %}{% block content %}
<div class="card">
  <h1 id="job-title">{{ job.title }}</h1>
  <p class="muted">{{ job.company }} · {{ job.location }} · {{ job.remote }}
     · ref {{ job.job_id }} · posted {{ job.posted_at }}</p>
  <pre id="job-description">{{ job.description }}</pre>
  <form method="post" action="/jobs/{{ job.job_id }}/apply">
    <button type="submit" id="apply-button">Apply for this position</button>
  </form>
</div>
{% endblock %}
"""

_STEP1 = """
{% extends "base" %}{% block content %}
<div class="card">
  <div class="steps"><b>1 Personal</b><span>2 Experience</span><span>3 Resume</span>
    <span>4 Questions</span><span>5 Review</span></div>
  <h2>Personal information</h2>
  {% if errors %}<div class="err" role="alert">{{ errors|join(" · ") }}</div>{% endif %}
  <form method="post">
    <label for="full_name">Full name *</label>
    <input type="text" id="full_name" name="full_name" required
           value="{{ d.full_name or '' }}">
    <div class="row">
      <div>
        <label for="email">Email address *</label>
        <input type="email" id="email" name="email" required value="{{ d.email or '' }}">
      </div>
      <div>
        <label for="phone">Phone *</label>
        <input type="tel" id="phone" name="phone" value="{{ d.phone or '' }}">
      </div>
    </div>
    <!-- Two deliberate awkwardnesses here: the label is not associated via `for`, and
         the field is marked required only by the "*" in that label, with validation
         happening server-side. Both are extremely common in real forms. -->
    <label>City of residence *</label>
    <input type="text" name="city" value="{{ d.city or '' }}">
    <label for="linkedin">LinkedIn profile URL</label>
    <input type="text" id="linkedin" name="linkedin" value="{{ d.linkedin or '' }}">
    <p style="margin-top:18px"><button type="submit" id="continue-1">Continue</button></p>
  </form>
</div>
{% endblock %}
"""

_STEP2 = """
{% extends "base" %}{% block content %}
<div class="card">
  <div class="steps"><span>1 Personal</span><b>2 Experience</b><span>3 Resume</span>
    <span>4 Questions</span><span>5 Review</span></div>
  <h2>Most recent experience</h2>
  {% if errors %}<div class="err" role="alert">{{ errors|join(" · ") }}</div>{% endif %}
  <form method="post">
    <label for="current_title">Current or most recent job title *</label>
    <input type="text" id="current_title" name="current_title" value="{{ d.current_title or '' }}">
    <label for="current_company">Company *</label>
    <input type="text" id="current_company" name="current_company"
           value="{{ d.current_company or '' }}">
    <label for="years_experience">Total years of professional experience *</label>
    <input type="number" step="0.5" min="0" id="years_experience" name="years_experience"
           value="{{ d.years_experience or '' }}">
    <label for="seniority">Seniority level *</label>
    <select id="seniority" name="seniority">
      <option value="">— Select —</option>
      {% for opt in ['intern','junior','mid','senior','lead','manager'] %}
      <option value="{{ opt }}" {% if d.seniority == opt %}selected{% endif %}>{{ opt|title }}</option>
      {% endfor %}
    </select>
    <p style="margin-top:18px">
      <button type="submit" name="action" value="back" class="secondary">Back</button>
      <button type="submit" name="action" value="next" id="continue-2">Continue</button>
    </p>
  </form>
</div>
{% endblock %}
"""

_STEP3 = """
{% extends "base" %}{% block content %}
<div class="card">
  <div class="steps"><span>1 Personal</span><span>2 Experience</span><b>3 Resume</b>
    <span>4 Questions</span><span>5 Review</span></div>
  <h2>Upload your resume</h2>
  {% if errors %}<div class="err" role="alert">{{ errors|join(" · ") }}</div>{% endif %}
  {% if app.resume_filename %}
    <p class="ok" id="upload-status">Uploaded: {{ app.resume_filename }}
       ({{ app.resume_size }} bytes)</p>
  {% endif %}
  <form method="post" enctype="multipart/form-data">
    <label for="resume">Resume file (PDF or DOCX, max 5 MB) *</label>
    <input type="file" id="resume" name="resume" accept=".pdf,.docx">
    <p style="margin-top:18px">
      <button type="submit" name="action" value="back" class="secondary">Back</button>
      <button type="submit" name="action" value="next" id="continue-3">Continue</button>
    </p>
  </form>
</div>
{% endblock %}
"""

_STEP4 = """
{% extends "base" %}{% block content %}
<div class="card">
  <div class="steps"><span>1 Personal</span><span>2 Experience</span><span>3 Resume</span>
    <b>4 Questions</b><span>5 Review</span></div>
  <h2>Screening questions</h2>
  {% if errors %}<div class="err" role="alert">{{ errors|join(" · ") }}</div>{% endif %}
  <form method="post">
    <label for="sql_years">How many years of experience do you have with SQL? *</label>
    <input type="number" step="0.5" min="0" id="sql_years" name="sql_years"
           value="{{ d.sql_years or '' }}">

    <label for="english_level">English proficiency *</label>
    <select id="english_level" name="english_level">
      <option value="">— Select —</option>
      {% for opt in ['basic','intermediate','advanced','fluent','native'] %}
      <option value="{{ opt }}" {% if d.english_level == opt %}selected{% endif %}>
        {{ opt|title }}</option>
      {% endfor %}
    </select>

    <fieldset>
      <legend>Are you legally authorized to work in Brazil? *</legend>
      <label style="font-weight:400">
        <input type="radio" name="work_auth" value="yes"
          {% if d.work_auth == 'yes' %}checked{% endif %}> Yes</label>
      <label style="font-weight:400">
        <input type="radio" name="work_auth" value="no"
          {% if d.work_auth == 'no' %}checked{% endif %}> No</label>
    </fieldset>

    <fieldset>
      <legend>Which of these have you used professionally?</legend>
      {% for tool in ['Grafana','Power BI','Tableau','Prometheus','dbt','Airflow'] %}
      <label style="font-weight:400">
        <input type="checkbox" name="tools" value="{{ tool }}"
          {% if tool in (d.tools or []) %}checked{% endif %}> {{ tool }}</label>
      {% endfor %}
    </fieldset>

    <label for="motivation">Why do you want to work at {{ job.company }}? *</label>
    <textarea id="motivation" name="motivation">{{ d.motivation or '' }}</textarea>

    <label style="font-weight:400; margin-top:16px">
      <input type="checkbox" name="consent" value="yes" {% if d.consent %}checked{% endif %}>
      I consent to the processing of my personal data for this recruitment process. *
    </label>

    <p style="margin-top:18px">
      <button type="submit" name="action" value="back" class="secondary">Back</button>
      <button type="submit" name="action" value="next" id="continue-4">Continue</button>
    </p>
  </form>
</div>
{% endblock %}
"""

_STEP5 = """
{% extends "base" %}{% block content %}
<div class="card">
  <div class="steps"><span>1 Personal</span><span>2 Experience</span><span>3 Resume</span>
    <span>4 Questions</span><b>5 Review</b></div>
  <h2>Review your application</h2>
  <p class="muted">{{ job.title }} — {{ job.company }}</p>
  <table style="width:100%; border-collapse:collapse" id="review-table">
    {% for key, value in rows %}
    <tr><td style="padding:6px 8px; color:#6b7280; width:45%">{{ key }}</td>
        <td style="padding:6px 8px"><b>{{ value }}</b></td></tr>
    {% endfor %}
  </table>
  <form method="post" style="margin-top:18px">
    <button type="submit" name="action" value="back" class="secondary">Back</button>
    <button type="submit" name="action" value="submit" id="submit-application">
      Submit application</button>
  </form>
</div>
{% endblock %}
"""

_DONE = """
{% extends "base" %}{% block content %}
<div class="card">
  <div class="ok" id="submission-confirmation">
    <h2 style="margin-top:0">Application submitted</h2>
    <p>Thank you for applying to <b>{{ job.title }}</b> at <b>{{ job.company }}</b>.</p>
    <p>Your reference number is <b id="reference-number">{{ app.reference }}</b>.</p>
  </div>
</div>
{% endblock %}
"""

_ENV = Environment(
    loader=DictLoader(
        {
            "base": _BASE,
            "jobs": _JOB_LIST,
            "job": _JOB_DETAIL,
            "step1": _STEP1,
            "step2": _STEP2,
            "step3": _STEP3,
            "step4": _STEP4,
            "step5": _STEP5,
            "done": _DONE,
        }
    ),
    autoescape=True,
)


def _render(template: str, **context: Any) -> HTMLResponse:
    context.setdefault("title", "MockATS")
    return HTMLResponse(_ENV.get_template(template).render(**context))


def create_app() -> FastAPI:
    app = FastAPI(title="MockATS", docs_url=None, redoc_url=None)

    def _app_or_404(token: str) -> MockApplication:
        application = APPLICATIONS.get(token)
        if application is None:
            raise KeyError(token)
        return application

    @app.get("/", response_class=HTMLResponse)
    @app.get("/jobs", response_class=HTMLResponse)
    async def list_jobs() -> HTMLResponse:
        return _render("jobs", jobs=list(MOCK_JOBS.values()), title="Open positions")

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail(job_id: str) -> HTMLResponse:
        job = MOCK_JOBS.get(job_id)
        if job is None:
            return HTMLResponse("<h1>Not found</h1>", status_code=404)
        return _render("job", job=job, title=job.title)

    @app.post("/jobs/{job_id}/apply")
    async def start_application(job_id: str) -> RedirectResponse:
        if job_id not in MOCK_JOBS:
            return RedirectResponse("/jobs", status_code=303)
        token = secrets.token_urlsafe(9)
        APPLICATIONS[token] = MockApplication(token=token, job_id=job_id)
        return RedirectResponse(f"/apply/{token}/1", status_code=303)

    @app.get("/apply/{token}/{step}", response_class=HTMLResponse)
    async def show_step(token: str, step: int) -> HTMLResponse:
        try:
            application = _app_or_404(token)
        except KeyError:
            return HTMLResponse("<h1>Application session expired</h1>", status_code=404)
        job = MOCK_JOBS[application.job_id]
        if application.submitted:
            return _render("done", app=application, job=job, title="Submitted")
        if step == 5:
            return _render("step5", app=application, job=job, rows=_review_rows(application),
                           title="Review")
        return _render(f"step{step}", app=application, job=job, d=application.data,
                       errors=[], title=f"Step {step}")

    @app.post("/apply/{token}/1", response_class=HTMLResponse)
    async def submit_step1(
        token: str,
        full_name: str = Form(""),
        email: str = Form(""),
        phone: str = Form(""),
        city: str = Form(""),
        linkedin: str = Form(""),
    ) -> Any:
        application = _app_or_404(token)
        payload = {
            "full_name": full_name.strip(),
            "email": email.strip(),
            "phone": phone.strip(),
            "city": city.strip(),
            "linkedin": linkedin.strip(),
        }
        errors = [
            f"{label} is required"
            for key, label in (
                ("full_name", "Full name"),
                ("email", "Email address"),
                ("phone", "Phone"),
                ("city", "City of residence"),
            )
            if not payload[key]
        ]
        if payload["email"] and "@" not in payload["email"]:
            errors.append("Email address is not valid")
        application.data.update(payload)
        if errors:
            return _render(
                "step1", app=application, job=MOCK_JOBS[application.job_id],
                d=application.data, errors=errors, title="Step 1",
            )
        application.step = 2
        return RedirectResponse(f"/apply/{token}/2", status_code=303)

    @app.post("/apply/{token}/2", response_class=HTMLResponse)
    async def submit_step2(
        token: str,
        action: str = Form("next"),
        current_title: str = Form(""),
        current_company: str = Form(""),
        years_experience: str = Form(""),
        seniority: str = Form(""),
    ) -> Any:
        application = _app_or_404(token)
        application.data.update(
            {
                "current_title": current_title.strip(),
                "current_company": current_company.strip(),
                "years_experience": years_experience.strip(),
                "seniority": seniority.strip(),
            }
        )
        if action == "back":
            return RedirectResponse(f"/apply/{token}/1", status_code=303)
        errors = [
            f"{label} is required"
            for key, label in (
                ("current_title", "Current or most recent job title"),
                ("current_company", "Company"),
                ("years_experience", "Total years of professional experience"),
                ("seniority", "Seniority level"),
            )
            if not application.data.get(key)
        ]
        if errors:
            return _render(
                "step2", app=application, job=MOCK_JOBS[application.job_id],
                d=application.data, errors=errors, title="Step 2",
            )
        application.step = 3
        return RedirectResponse(f"/apply/{token}/3", status_code=303)

    @app.post("/apply/{token}/3", response_class=HTMLResponse)
    async def submit_step3(
        request: Request,
        token: str,
        action: str = Form("next"),
        resume: UploadFile | None = None,
    ) -> Any:
        application = _app_or_404(token)
        if action == "back":
            return RedirectResponse(f"/apply/{token}/2", status_code=303)

        errors: list[str] = []
        if resume is not None and resume.filename:
            content = await resume.read()
            if not resume.filename.lower().endswith(ALLOWED_UPLOAD_SUFFIXES):
                errors.append("Resume must be a PDF or DOCX file")
            elif len(content) > MAX_UPLOAD_BYTES:
                errors.append("Resume exceeds the 5 MB limit")
            elif not content:
                errors.append("Uploaded file is empty")
            else:
                application.resume_filename = resume.filename
                application.resume_size = len(content)
        if not application.resume_filename:
            errors.append("A resume file is required")

        if errors:
            return _render(
                "step3", app=application, job=MOCK_JOBS[application.job_id],
                d=application.data, errors=errors, title="Step 3",
            )
        application.step = 4
        return RedirectResponse(f"/apply/{token}/4", status_code=303)

    @app.post("/apply/{token}/4", response_class=HTMLResponse)
    async def submit_step4(request: Request, token: str) -> Any:
        application = _app_or_404(token)
        form = await request.form()
        action = str(form.get("action", "next"))
        application.data.update(
            {
                "sql_years": str(form.get("sql_years", "")).strip(),
                "english_level": str(form.get("english_level", "")).strip(),
                "work_auth": str(form.get("work_auth", "")).strip(),
                "tools": [str(v) for v in form.getlist("tools")],
                "motivation": str(form.get("motivation", "")).strip(),
                "consent": bool(form.get("consent")),
            }
        )
        if action == "back":
            return RedirectResponse(f"/apply/{token}/3", status_code=303)

        data = application.data
        errors = []
        if not data.get("sql_years"):
            errors.append("Years of SQL experience is required")
        if not data.get("english_level"):
            errors.append("English proficiency is required")
        if not data.get("work_auth"):
            errors.append("Work authorization answer is required")
        if len(str(data.get("motivation", ""))) < 40:
            errors.append("Motivation must be at least 40 characters")
        if not data.get("consent"):
            errors.append("You must consent to data processing to continue")

        if errors:
            return _render(
                "step4", app=application, job=MOCK_JOBS[application.job_id],
                d=application.data, errors=errors, title="Step 4",
            )
        application.step = 5
        return RedirectResponse(f"/apply/{token}/5", status_code=303)

    @app.post("/apply/{token}/5", response_class=HTMLResponse)
    async def submit_final(token: str, action: str = Form("submit")) -> Any:
        application = _app_or_404(token)
        if action == "back":
            return RedirectResponse(f"/apply/{token}/4", status_code=303)
        application.submitted = True
        application.reference = f"MA-{secrets.randbelow(900000) + 100000}"
        return RedirectResponse(f"/apply/{token}/done", status_code=303)

    @app.get("/apply/{token}/done", response_class=HTMLResponse)
    async def done(token: str) -> HTMLResponse:
        application = _app_or_404(token)
        return _render(
            "done", app=application, job=MOCK_JOBS[application.job_id], title="Submitted"
        )

    @app.get("/_debug/applications")
    async def debug_applications() -> dict[str, Any]:
        """Test hook: lets the test suite assert what the ATS actually received."""
        return {
            token: {
                "job_id": a.job_id,
                "submitted": a.submitted,
                "reference": a.reference,
                "resume_filename": a.resume_filename,
                "data": a.data,
            }
            for token, a in APPLICATIONS.items()
        }

    @app.post("/_debug/reset")
    async def debug_reset() -> dict[str, str]:
        APPLICATIONS.clear()
        return {"status": "reset"}

    return app


def _review_rows(application: MockApplication) -> list[tuple[str, str]]:
    data = application.data
    return [
        ("Full name", str(data.get("full_name", ""))),
        ("Email", str(data.get("email", ""))),
        ("Phone", str(data.get("phone", ""))),
        ("City", str(data.get("city", ""))),
        ("Current title", str(data.get("current_title", ""))),
        ("Current company", str(data.get("current_company", ""))),
        ("Years of experience", str(data.get("years_experience", ""))),
        ("Seniority", str(data.get("seniority", ""))),
        ("Resume", application.resume_filename or "(none)"),
        ("SQL years", str(data.get("sql_years", ""))),
        ("English", str(data.get("english_level", ""))),
        ("Work authorization", str(data.get("work_auth", ""))),
        ("Tools", ", ".join(data.get("tools", []) or []) or "(none)"),
        ("Motivation", str(data.get("motivation", ""))[:300]),
    ]


app = create_app()

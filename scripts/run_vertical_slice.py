"""End-to-end vertical slice against the local MockATS.

Exercises the real pipeline with the real local model and a real Chromium:

    profile -> search -> discovery -> job intelligence -> match -> ATS analysis
    -> resume optimization -> fact validation -> render -> browser agent
    -> multi-step form -> upload -> pre-submission validation -> submit
    -> post-submission validation -> history -> artifact cleanup

Auto-submit is enabled here *because the target is a local mock server*. Nothing
leaves the machine. Run it with::

    python -m cvnator.mockats          # terminal 1
    python scripts/run_vertical_slice.py   # terminal 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvnator.agent.orchestrator import AgentSession  # noqa: E402
from cvnator.ai.gateway import get_gateway  # noqa: E402
from cvnator.config import get_settings  # noqa: E402
from cvnator.db import repositories as repo  # noqa: E402
from cvnator.db.base import session_scope  # noqa: E402
from cvnator.domain.profile import (  # noqa: E402
    CandidateMasterProfile,
    Education,
    Experience,
    LanguageSkill,
    PersonalInformation,
    Proficiency,
    Project,
    RemotePolicy,
    SalaryPreferences,
    Skill,
    WorkAuthorization,
    WorkPreferences,
)
from cvnator.domain.search import AutonomySettings, SearchProfile  # noqa: E402
from cvnator.observability import setup_logging  # noqa: E402
from cvnator.platforms import generic as _generic  # noqa: F401,E402
from cvnator.platforms import mock as _mock  # noqa: F401,E402

MOCK_URL = "http://127.0.0.1:8099"


def build_profile() -> CandidateMasterProfile:
    """A realistic candidate. In production this comes from the CV parser."""
    return CandidateMasterProfile(
        personal_information=PersonalInformation(
            full_name="Lucas Ismael",
            email="lucas.ismael@example.com",
            phone="+55 11 98888-7777",
            city="São Paulo",
            state="SP",
            country="Brasil",
            linkedin="https://linkedin.com/in/lucasismael",
            github="https://github.com/lucasismael",
        ),
        headline="Data Analyst focused on operational reporting and observability",
        professional_summary=(
            "Data analyst with experience building operational dashboards, modelling data "
            "in SQL and automating extraction routines in Python."
        ),
        experiences=[
            Experience(
                company="Northbridge Logística",
                job_title="Data Analyst",
                start_date="2022-03",
                end_date="",
                is_current=True,
                location="São Paulo, BR",
                technologies=["Python", "SQL", "Grafana", "PostgreSQL"],
                responsibilities=[
                    "Criação de dashboards operacionais para acompanhamento de indicadores.",
                    "Modelagem de dados em SQL para relatórios recorrentes.",
                    "Automação de rotinas de extração de dados em Python.",
                ],
                achievements=[
                    "Redução de 30% no tempo de resposta a incidentes operacionais.",
                ],
            ),
            Experience(
                company="Vertex Serviços",
                job_title="Analista de Suporte de Dados",
                start_date="2020-06",
                end_date="2022-02",
                location="Campinas, BR",
                technologies=["SQL", "Excel"],
                responsibilities=[
                    "Atendimento a solicitações de extração de dados para áreas de negócio.",
                    "Manutenção de planilhas de controle operacional.",
                ],
                achievements=[],
            ),
        ],
        projects=[
            Project(
                name="Painel de SLA",
                description="Painel interno de acompanhamento de SLA de atendimento.",
                technologies=["Grafana", "PostgreSQL"],
                highlights=["Consolidou três relatórios manuais em um painel único."],
            )
        ],
        education=[
            Education(
                institution="Universidade de São Paulo",
                degree="Bacharelado",
                field_of_study="Sistemas de Informação",
                start_date="2017",
                end_date="2021",
                status="completed",
            )
        ],
        languages=[
            LanguageSkill(language="Português", proficiency=Proficiency.NATIVE),
            LanguageSkill(language="Inglês", proficiency=Proficiency.ADVANCED),
        ],
        skills=[
            Skill(name="Python", category="hard", years_of_experience=4.0),
            Skill(name="SQL", category="hard", years_of_experience=5.0),
            Skill(name="Grafana", category="hard", years_of_experience=3.0),
            Skill(name="PostgreSQL", category="hard"),
            Skill(name="Excel", category="hard"),
            Skill(name="Data Visualization", category="hard"),
        ],
        salary_preferences=SalaryPreferences(currency="BRL", minimum=9000, target=11000),
        work_preferences=WorkPreferences(remote_policy=RemotePolicy.REMOTE),
        work_authorization=WorkAuthorization(
            countries_authorized=["Brasil"], requires_sponsorship=False
        ),
    )


def build_search() -> SearchProfile:
    return SearchProfile(
        name="Data Analyst (mock)",
        role_title="Data Analyst",
        alternate_titles=["Analista de Dados", "Data Analytics"],
        keywords=["dashboard", "SQL", "Python"],
        technologies=["SQL", "Python", "Grafana"],
        remote_policies=[RemotePolicy.ANY],
        platforms=["mock"],
        minimum_match_score=0.45,
        daily_application_limit=5,
    )


async def ensure_mock_running() -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{MOCK_URL}/jobs")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"MockATS is not reachable at {MOCK_URL} ({exc}).\n"
                "Start it first:  python -m cvnator.mockats"
            ) from exc
        await client.post(f"{MOCK_URL}/_debug/reset")


async def main() -> dict:
    setup_logging("INFO")
    settings = get_settings()
    await ensure_mock_running()
    started_at = time.monotonic()

    profile = build_profile()
    search = build_search()

    with session_scope() as session:
        repo.save_profile(session, profile)
        search = repo.save_search(session, search)

    autonomy = AutonomySettings(
        auto_search=True,
        auto_match=True,
        auto_optimize_resume=True,
        auto_apply=True,
        # Enabled deliberately: the only target is a local mock server.
        auto_submit=True,
        review_before_submit=False,
        dry_run=False,
        delete_temp_resumes=True,
        minimum_match_score=0.45,
        daily_application_limit=2,
        session_application_limit=2,
    )

    events: list[tuple[str, dict]] = []
    session_obj = AgentSession(
        profile,
        [search],
        autonomy,
        gateway=get_gateway(),
        settings=settings,
        event_sink=lambda kind, payload: events.append((kind, payload)),
    )
    # Keep the slice quick: only the first posting is needed to prove the path.
    session_obj.rate_limiter.default.min_application_gap_s = 1.0
    session_obj.rate_limiter.default.max_application_gap_s = 2.0

    print("\n=== Starting session ===")
    await session_obj.start()
    assert session_obj._task is not None
    try:
        await asyncio.wait_for(session_obj._task, timeout=1500)
    except TimeoutError:
        print("!! Session exceeded the time budget; stopping")
        await session_obj.stop()

    print("\n=== Metrics ===")
    print(json.dumps(session_obj.metrics.as_dict(), indent=2))

    print("\n=== Notable events ===")
    for kind, payload in events:
        if kind in {
            "resume_generated", "application_submitted", "application_waiting",
            "artifacts_cleaned", "user_action_required", "error",
        }:
            print(f"  {kind}: {json.dumps(payload, ensure_ascii=False)[:300]}")

    print("\n=== What the MockATS actually received ===")
    async with httpx.AsyncClient(timeout=10.0) as client:
        received = (await client.get(f"{MOCK_URL}/_debug/applications")).json()
    submitted = {k: v for k, v in received.items() if v["submitted"]}
    print(json.dumps(submitted, indent=2, ensure_ascii=False)[:2500])

    print("\n=== Application history ===")
    applications = []
    with session_scope() as session:
        for application in repo.list_applications(session, limit=10):
            print(
                f"  [{application['state']}] {application['job']['role']} @ "
                f"{application['job']['company']} | match="
                f"{application['job']['match_score']} | "
                f"checkpoints={application['checkpoints']} | "
                f"errors={application['errors']}"
            )
            applications.append(
                {
                    "state": application["state"],
                    "role": application["job"]["role"],
                    "company": application["job"]["company"],
                    "match_score": application["job"]["match_score"],
                    "checkpoints": len(application["checkpoints"]),
                    "errors": application["errors"],
                }
            )

    print("\n=== Temporary resume files remaining ===")
    leftovers = list(settings.temp_resumes_dir.glob("application_*"))
    print(f"  {len(leftovers)} file(s): {[p.name for p in leftovers]}")

    await get_gateway().aclose()

    return {
        "succeeded": bool(submitted),
        "wall_seconds": round(time.monotonic() - started_at, 1),
        "metrics": session_obj.metrics.as_dict(),
        "submitted": {
            token: {
                "job_id": record.get("job_id"),
                "reference": record.get("reference"),
                "fields": len(record.get("data", {})),
            }
            for token, record in submitted.items()
        },
        "applications": applications,
        "user_action_requests": [
            payload for kind, payload in events if kind == "user_action_required"
        ],
        "errors": [payload for kind, payload in events if kind == "error"],
        "leftover_temp_files": [p.name for p in leftovers],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write a machine-readable summary of the run to this path.",
    )
    args = parser.parse_args()
    outcome = asyncio.run(main())
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    raise SystemExit(0 if outcome["succeeded"] else 1)

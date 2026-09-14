"""Shared fixtures.

Every test runs against a throwaway SQLite file and a throwaway data directory, so
the suite never touches the developer's real profile, resumes or history.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    data_dir = tmp_path_factory.mktemp("cvnator-data")
    os.environ["CVNATOR_DATA_DIR"] = str(data_dir)
    os.environ["CVNATOR_DB_URL"] = f"sqlite+pysqlite:///{(data_dir / 'test.sqlite3').as_posix()}"
    os.environ["CVNATOR_BROWSER_HEADLESS"] = "true"

    from cvnator.config import get_settings

    get_settings.cache_clear()
    get_settings()
    yield
    get_settings.cache_clear()


@pytest.fixture
def db_session(_isolated_settings: None) -> Iterator:
    """A session against freshly created tables, rolled back by dropping them."""
    from cvnator.db import models  # noqa: F401  (registers tables)
    from cvnator.db.base import Base, get_engine, get_session_factory

    engine = get_engine()
    Base.metadata.create_all(engine)
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def profile():
    from cvnator.domain.profile import (
        CandidateMasterProfile,
        Education,
        Experience,
        LanguageSkill,
        PersonalInformation,
        Proficiency,
        Skill,
        WorkAuthorization,
    )

    return CandidateMasterProfile(
        personal_information=PersonalInformation(
            full_name="Ana Souza",
            email="ana@example.com",
            phone="+55 11 90000-0000",
            city="São Paulo",
            country="Brasil",
        ),
        professional_summary="Analista de dados focada em relatórios operacionais.",
        experiences=[
            Experience(
                company="Acme Log",
                job_title="Data Analyst",
                start_date="2022-03",
                end_date="",
                is_current=True,
                technologies=["Python", "SQL", "Grafana"],
                responsibilities=["Criação de dashboards operacionais."],
                achievements=["Redução de 30% no tempo de resposta a incidentes."],
            ),
        ],
        education=[
            Education(institution="USP", degree="Bacharelado", field_of_study="SI",
                      end_date="2021")
        ],
        languages=[
            LanguageSkill(language="Português", proficiency=Proficiency.NATIVE),
            LanguageSkill(language="Inglês", proficiency=Proficiency.ADVANCED),
        ],
        skills=[
            Skill(name="Python", category="hard", years_of_experience=3.0),
            Skill(name="SQL", category="hard"),
            Skill(name="Grafana", category="hard"),
        ],
        work_authorization=WorkAuthorization(countries_authorized=["Brasil"]),
    )


@pytest.fixture
def job():
    from cvnator.domain.job import ApplyMethod, JobPosting, JobRequirement
    from cvnator.domain.profile import RemotePolicy

    return JobPosting(
        platform="mock",
        platform_job_id="MJ-1001",
        url="http://127.0.0.1:8099/jobs/MJ-1001",
        company="Northwind Analytics",
        role="Data Analyst",
        location="São Paulo, BR",
        remote_policy=RemotePolicy.REMOTE,
        description=(
            "Build dashboards for operational KPIs. Model data in SQL. "
            "Requirements: 2+ years of SQL, Python, Grafana. Preferred: dbt, AWS."
        ),
        requirements=[
            JobRequirement(text="2+ years of SQL", is_required=True),
            JobRequirement(text="Python", is_required=True),
            JobRequirement(text="dbt", is_required=False),
        ],
        apply_method=ApplyMethod.NATIVE_FORM,
    )


@pytest.fixture
def search():
    from cvnator.domain.profile import RemotePolicy
    from cvnator.domain.search import SearchProfile

    return SearchProfile(
        name="Data Analyst",
        role_title="Data Analyst",
        keywords=["dashboard", "SQL"],
        technologies=["SQL", "Python", "Grafana"],
        remote_policies=[RemotePolicy.ANY],
        platforms=["mock"],
        minimum_match_score=0.5,
    )


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A real, text-based PDF produced by the renderer, for extraction tests."""
    from cvnator.resume.optimizer import RenderedExperience, ResumeDocument
    from cvnator.resume.renderer import render_pdf

    document = ResumeDocument(
        full_name="Ana Souza",
        contact_lines=["ana@example.com", "+55 11 90000-0000"],
        headline="Data Analyst",
        professional_summary="Analista de dados focada em relatórios operacionais.",
        experiences=[
            RenderedExperience(
                company="Acme Log", job_title="Data Analyst", start_date="2022-03",
                end_date="", is_current=True, location="São Paulo",
                bullets=["Criação de dashboards operacionais."],
                technologies=["Python", "SQL"],
            )
        ],
        skills=["Python", "SQL", "Grafana"],
        projects=[],
        education=["Bacharelado | SI | USP"],
        certifications=[],
        languages=["Português (native)"],
    )
    path = tmp_path / "cv.pdf"
    render_pdf(document, path, language="pt")
    return path

"""The post-submission cleanup path.

Deleting the tailored CV is only safe because the recipe survives it. These tests
pin both halves of that bargain: the file goes, the `ResumeVersion` row stays and is
marked as having had its file removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from cvnator.agent.orchestrator import AgentSession
from cvnator.db.models import ApplicationRow, ArtifactRow, JobRow, ResumeVersionRow
from cvnator.domain.search import AutonomySettings
from cvnator.storage.artifacts import ArtifactKind, ArtifactStore


@pytest.fixture
def session_under_test(profile, search, db_session, monkeypatch):
    """An AgentSession wired to the test database, with no browser and no models."""
    from contextlib import contextmanager

    # session_scope() opens its own session; point it at the one the fixture set up
    # so the rows created here are visible to the code under test.
    import cvnator.agent.orchestrator as orchestrator_module
    from cvnator.ai.gateway import AIGateway
    from cvnator.db.base import get_session_factory

    @contextmanager
    def _scope():
        yield db_session
        db_session.flush()

    monkeypatch.setattr(orchestrator_module, "session_scope", _scope)
    assert get_session_factory() is not None

    return AgentSession(
        profile,
        [search],
        AutonomySettings(dry_run=False, delete_temp_resumes=True),
        gateway=AIGateway(),
    )


def _prepare_application(db_session, tmp_path: Path) -> tuple[int, Path, str]:
    job = JobRow(platform="mock", platform_job_id="MJ-CLEAN", company="X", role="Data Analyst")
    db_session.add(job)
    db_session.flush()

    application = ApplicationRow(job_id=job.id, platform="mock", state="SUBMITTING")
    db_session.add(application)
    db_session.flush()

    pdf = tmp_path / "application_abc123.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    docx = tmp_path / "application_abc123.docx"
    docx.write_bytes(b"fake docx")

    store = ArtifactStore(db_session)
    store.register(pdf, ArtifactKind.TEMP_RESUME, in_use=True)
    store.register(docx, ArtifactKind.INTERMEDIATE)

    db_session.add(
        ResumeVersionRow(
            resume_version_id="abc123",
            source_profile_version=1,
            job_id=job.id,
            job_description_hash="hash",
            template_used="ats_plain_pdf",
            transformation_summary={"headline": "Data Analyst"},
            optimized_document={"skills": ["SQL"]},
            ats_score_before=0.74,
            ats_score_after=0.82,
            file_sha256="deadbeef",
            file_path=str(pdf),
        )
    )
    db_session.flush()
    return application.id, pdf, "abc123"


def test_confirmed_submission_deletes_the_file_but_keeps_the_recipe(
    session_under_test, db_session, tmp_path
):
    application_id, pdf, version_id = _prepare_application(db_session, tmp_path)
    docx = pdf.with_suffix(".docx")

    session_under_test._cleanup_resume(application_id, pdf, version_id)

    assert not pdf.exists(), "the tailored PDF should be gone"
    assert not docx.exists(), "the intermediate DOCX should be gone"

    recipe = db_session.execute(
        select(ResumeVersionRow).where(ResumeVersionRow.resume_version_id == version_id)
    ).scalar_one()
    assert recipe.file_deleted is True
    # Everything needed to rebuild the identical document must survive.
    assert recipe.ats_score_before == 0.74
    assert recipe.ats_score_after == 0.82
    assert recipe.transformation_summary["headline"] == "Data Analyst"
    assert recipe.file_sha256 == "deadbeef"
    assert recipe.template_used == "ats_plain_pdf"


def test_cleanup_marks_the_artifacts_deleted_in_the_ledger(
    session_under_test, db_session, tmp_path
):
    application_id, pdf, version_id = _prepare_application(db_session, tmp_path)
    session_under_test._cleanup_resume(application_id, pdf, version_id)

    rows = list(
        db_session.execute(
            select(ArtifactRow).where(ArtifactRow.path == str(pdf.resolve()))
        ).scalars()
    )
    assert rows and rows[0].deleted
    assert rows[0].deleted_at is not None


def test_a_failed_application_releases_the_hold_without_deleting(
    session_under_test, db_session, tmp_path
):
    """A failure keeps the file: the user may want to inspect or retry it."""
    _application_id, pdf, _version = _prepare_application(db_session, tmp_path)

    session_under_test._release_resume_hold(pdf)

    assert pdf.exists()
    row = db_session.execute(
        select(ArtifactRow).where(ArtifactRow.path == str(pdf.resolve()))
    ).scalar_one()
    assert row.in_use is False
    assert row.deleted is False


def test_cleanup_is_a_no_op_without_a_resume(session_under_test, db_session, tmp_path):
    application_id, _pdf, _version = _prepare_application(db_session, tmp_path)
    session_under_test._cleanup_resume(application_id, None, "abc123")  # must not raise


def test_the_session_limit_counts_submissions_not_attempts(session_under_test) -> None:
    """A limit of N must deliver N applications, not N-1.

    Counting *started* applications blocked the Nth application from submitting by
    virtue of its own start: with a limit of 2 the agent filled both forms, reached
    the review screen on the second, and abandoned it there.
    """
    session = session_under_test
    session.autonomy.session_application_limit = 2

    session.metrics.applications_started = 1
    assert session.submissions_this_session == 0

    # Second application started, first one submitted: still under the limit.
    session.metrics.applications_started = 2
    session.metrics.applications_completed = 1
    assert session.submissions_this_session < session.autonomy.session_application_limit

    session.metrics.applications_completed = 2
    assert session.submissions_this_session == session.autonomy.session_application_limit

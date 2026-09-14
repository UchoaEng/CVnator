"""Repository helpers: the only place that translates between ORM rows and domain models.

Keeping the mapping here means the API, the orchestrator and the agent all work with
validated pydantic objects and never with detached SQLAlchemy instances.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from cvnator.db.models import (
    AgentSessionRow,
    ApplicationRow,
    CandidateProfileRow,
    JobRow,
    SearchProfileRow,
    UserActionRow,
)
from cvnator.domain.profile import CandidateMasterProfile
from cvnator.domain.search import SearchProfile
from cvnator.observability import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------ candidate profile


def save_profile(
    session: Session,
    profile: CandidateMasterProfile,
    *,
    master_cv_path: str = "",
    master_cv_sha256: str = "",
) -> CandidateProfileRow:
    """Store a new profile version and make it the active one.

    Versions are kept rather than overwritten, so a generated resume can always name
    the exact profile version it was built from.
    """
    for row in session.execute(
        select(CandidateProfileRow).where(CandidateProfileRow.is_active.is_(True))
    ).scalars():
        row.is_active = False

    latest = session.execute(
        select(CandidateProfileRow).order_by(CandidateProfileRow.version.desc())
    ).scalars().first()
    version = (latest.version + 1) if latest else 1
    profile.version = version

    row = CandidateProfileRow(
        version=version,
        is_active=True,
        full_name=profile.personal_information.full_name,
        email=profile.personal_information.email,
        payload=profile.model_dump(mode="json"),
        master_cv_path=master_cv_path or profile.master_cv_path,
        master_cv_sha256=master_cv_sha256,
    )
    session.add(row)
    session.flush()
    log.info("Saved candidate profile version %s", version)
    return row


def load_active_profile(session: Session) -> CandidateMasterProfile | None:
    row = session.execute(
        select(CandidateProfileRow).where(CandidateProfileRow.is_active.is_(True))
    ).scalars().first()
    if row is None:
        return None
    return CandidateMasterProfile.model_validate(row.payload)


# ---------------------------------------------------------------------- search profiles


def _search_from_row(row: SearchProfileRow) -> SearchProfile:
    payload = dict(row.payload or {})
    payload["id"] = row.id
    return SearchProfile.model_validate(payload)


def list_searches(session: Session, *, enabled_only: bool = False) -> list[SearchProfile]:
    stmt = select(SearchProfileRow).order_by(SearchProfileRow.priority.desc())
    if enabled_only:
        stmt = stmt.where(SearchProfileRow.enabled.is_(True))
    return [_search_from_row(row) for row in session.execute(stmt).scalars()]


def save_search(session: Session, search: SearchProfile) -> SearchProfile:
    payload = search.model_dump(mode="json")
    payload.pop("id", None)

    row = session.get(SearchProfileRow, search.id) if search.id else None
    if row is None:
        row = SearchProfileRow()
        session.add(row)

    row.name = search.name
    row.enabled = search.enabled
    row.role_title = search.role_title
    row.minimum_match_score = search.minimum_match_score
    row.daily_application_limit = search.daily_application_limit
    row.priority = search.priority
    row.payload = payload
    session.flush()
    return _search_from_row(row)


def delete_search(session: Session, search_id: int) -> bool:
    row = session.get(SearchProfileRow, search_id)
    if row is None:
        return False
    session.delete(row)
    return True


# ------------------------------------------------------------------------ applications


def list_applications(
    session: Session, *, limit: int = 100, state: str | None = None
) -> list[dict]:
    stmt = (
        select(ApplicationRow, JobRow)
        .join(JobRow, ApplicationRow.job_id == JobRow.id)
        .order_by(ApplicationRow.started_at.desc())
        .limit(limit)
    )
    if state:
        stmt = stmt.where(ApplicationRow.state == state)

    results: list[dict] = []
    for application, job in session.execute(stmt):
        results.append(
            {
                "id": application.id,
                "state": application.state,
                "platform": application.platform,
                "external_platform": application.external_platform,
                "dry_run": application.dry_run,
                "started_at": application.started_at.isoformat(),
                "completed_at": (
                    application.completed_at.isoformat() if application.completed_at else None
                ),
                "checkpoints": application.checkpoints,
                "errors": application.errors,
                "resume_version_id": application.resume_version_id,
                "submission_evidence": application.submission_evidence,
                "job": {
                    "id": job.id,
                    "role": job.role,
                    "company": job.company,
                    "location": job.location,
                    "url": job.url,
                    "match_score": job.match_score,
                    "match_report": job.match_report,
                },
            }
        )
    return results


def list_jobs(session: Session, *, limit: int = 200, status: str | None = None) -> list[dict]:
    stmt = select(JobRow).order_by(JobRow.date_discovered.desc()).limit(limit)
    if status:
        stmt = stmt.where(JobRow.status == status)
    return [
        {
            "id": row.id,
            "platform": row.platform,
            "role": row.role,
            "company": row.company,
            "location": row.location,
            "url": row.url,
            "status": row.status,
            "match_score": row.match_score,
            "match_report": row.match_report,
            "date_discovered": row.date_discovered.isoformat(),
        }
        for row in session.execute(stmt).scalars()
    ]


# ------------------------------------------------------------------------ user actions


def list_user_actions(session: Session, *, unresolved_only: bool = True) -> list[dict]:
    stmt = select(UserActionRow).order_by(UserActionRow.created_at.desc()).limit(100)
    if unresolved_only:
        stmt = stmt.where(UserActionRow.resolved.is_(False))
    return [
        {
            "id": row.id,
            "kind": row.kind,
            "platform": row.platform,
            "company": row.company,
            "job_title": row.job_title,
            "problem": row.problem,
            "action_needed": row.action_needed,
            "question": row.question,
            "resolved": row.resolved,
            "created_at": row.created_at.isoformat(),
        }
        for row in session.execute(stmt).scalars()
    ]


# ---------------------------------------------------------------------------- sessions


def open_session_row(session: Session, autonomy: dict) -> AgentSessionRow:
    row = AgentSessionRow(status="running", autonomy=autonomy, metrics={})
    session.add(row)
    session.flush()
    return row


def close_session_row(session: Session, session_id: int, metrics: dict) -> None:
    row = session.get(AgentSessionRow, session_id)
    if row is not None:
        row.status = "stopped"
        row.metrics = metrics
        row.stopped_at = datetime.now(UTC)


def interrupted_applications(session: Session) -> list[dict]:
    """Applications left mid-flight by a crash or a hard stop, for session recovery."""
    active = {"OPENING_APPLICATION", "APPLYING", "SUBMITTING", "WAITING_USER", "REVIEW_PENDING"}
    stmt = (
        select(ApplicationRow, JobRow)
        .join(JobRow, ApplicationRow.job_id == JobRow.id)
        .where(ApplicationRow.state.in_(active), ApplicationRow.completed_at.is_(None))
        .order_by(ApplicationRow.started_at.desc())
    )
    return [
        {
            "id": application.id,
            "state": application.state,
            "checkpoints": application.checkpoints,
            "resume_version_id": application.resume_version_id,
            "job": {"role": job.role, "company": job.company, "url": job.url,
                    "platform": job.platform},
        }
        for application, job in session.execute(stmt)
    ]

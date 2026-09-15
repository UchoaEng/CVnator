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
    AnalyticsEventRow,
    ApplicationRow,
    AppStateRow,
    ArtifactRow,
    CandidateProfileRow,
    JobRow,
    ResumeVersionRow,
    SearchProfileRow,
    StateTransitionRow,
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


# ------------------------------------------------------------------ application detail


def get_application(session: Session, application_id: int) -> dict | None:
    """Everything one application knows about itself, for the details page.

    The resume *recipe* is included even when its file has been deleted -- that is the
    point of keeping it (see ADR on temporary resumes): the user can still see which
    version was sent, what it scored and what changed.
    """
    row = session.get(ApplicationRow, application_id)
    if row is None:
        return None
    job = session.get(JobRow, row.job_id)

    transitions = (
        session.execute(
            select(StateTransitionRow)
            .where(StateTransitionRow.application_id == application_id)
            .order_by(StateTransitionRow.created_at)
        )
        .scalars()
        .all()
    )

    resume: dict | None = None
    if row.resume_version_id:
        version = session.execute(
            select(ResumeVersionRow).where(
                ResumeVersionRow.resume_version_id == row.resume_version_id
            )
        ).scalar_one_or_none()
        if version is not None:
            resume = {
                "resume_version_id": version.resume_version_id,
                "source_profile_version": version.source_profile_version,
                "template_used": version.template_used,
                "transformation_summary": version.transformation_summary,
                "ats_score_before": version.ats_score_before,
                "ats_score_after": version.ats_score_after,
                "file_deleted": version.file_deleted,
                "generation_date": version.generation_date.isoformat(),
            }

    return {
        "id": row.id,
        "state": row.state,
        "platform": row.platform,
        "external_platform": row.external_platform,
        "dry_run": row.dry_run,
        "started_at": row.started_at.isoformat(),
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "checkpoints": row.checkpoints,
        "errors": row.errors,
        "answers_used": row.answers_used,
        "decisions_summary": row.decisions_summary,
        "submission_evidence": row.submission_evidence,
        "job": {
            "id": job.id if job else None,
            "role": job.role if job else "",
            "company": job.company if job else "",
            "location": job.location if job else "",
            "url": job.url if job else "",
            "platform": job.platform if job else "",
            "description": (job.description[:4000] if job else ""),
            "match_score": job.match_score if job else None,
            "match_report": job.match_report if job else None,
            "ats_score": job.ats_score if job else None,
        },
        "resume": resume,
        "timeline": [
            {
                "from_state": t.from_state,
                "to_state": t.to_state,
                "reason": t.reason,
                "at": t.created_at.isoformat(),
            }
            for t in transitions
        ],
    }


# ------------------------------------------------------------------ documents


def list_documents(session: Session) -> dict[str, list[dict]]:
    """Files grouped the way the user thinks about them, not the way we store them.

    ``master`` is never touched by cleanup; ``temporary`` is what the garbage
    collector may remove; ``generated`` are recipes whose file may already be gone.
    """
    profile = session.execute(
        select(CandidateProfileRow).where(CandidateProfileRow.is_active.is_(True))
    ).scalar_one_or_none()

    master: list[dict] = []
    if profile is not None and profile.master_cv_path:
        from pathlib import Path

        path = Path(profile.master_cv_path)
        master.append(
            {
                "name": path.name,
                "path": str(path),
                "kind": "master_cv",
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "created_at": profile.created_at.isoformat(),
                "exists": path.exists(),
                "auto_delete": False,
            }
        )

    artifacts = (
        session.execute(
            select(ArtifactRow)
            .where(ArtifactRow.deleted.is_(False))
            .order_by(ArtifactRow.created_at.desc())
            .limit(500)
        )
        .scalars()
        .all()
    )
    temporary = [
        {
            "name": row.path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
            "path": row.path,
            "kind": row.kind,
            "size_bytes": row.size_bytes,
            "created_at": row.created_at.isoformat(),
            "application_id": row.application_id,
            "in_use": row.in_use,
            "exists": True,
            "auto_delete": not row.permanent,
        }
        for row in artifacts
        if not row.permanent
    ]

    versions = (
        session.execute(
            select(ResumeVersionRow).order_by(ResumeVersionRow.generation_date.desc()).limit(200)
        )
        .scalars()
        .all()
    )
    generated = [
        {
            "resume_version_id": row.resume_version_id,
            "job_id": row.job_id,
            "template_used": row.template_used,
            "ats_score_before": row.ats_score_before,
            "ats_score_after": row.ats_score_after,
            "created_at": row.generation_date.isoformat(),
            "file_deleted": row.file_deleted,
            "transformation_summary": row.transformation_summary,
        }
        for row in versions
    ]

    return {"master": master, "temporary": temporary, "generated": generated}


# ------------------------------------------------------------------ app state


def get_app_state(session: Session, key: str, default: dict | None = None) -> dict:
    row = session.execute(select(AppStateRow).where(AppStateRow.key == key)).scalar_one_or_none()
    if row is None:
        return dict(default or {})
    return dict(row.value)


def set_app_state(session: Session, key: str, value: dict) -> dict:
    row = session.execute(select(AppStateRow).where(AppStateRow.key == key)).scalar_one_or_none()
    if row is None:
        row = AppStateRow(key=key, value=value)
        session.add(row)
    else:
        row.value = value
    session.flush()
    return dict(row.value)


# ------------------------------------------------------------------ analytics


def record_event(session: Session, name: str, properties: dict | None = None) -> None:
    session.add(AnalyticsEventRow(name=name, properties=properties or {}))
    session.flush()


def list_events(session: Session, *, limit: int = 500) -> list[dict]:
    rows = (
        session.execute(
            select(AnalyticsEventRow).order_by(AnalyticsEventRow.created_at.desc()).limit(limit)
        )
        .scalars()
        .all()
    )
    return [
        {"name": r.name, "properties": r.properties, "at": r.created_at.isoformat()} for r in rows
    ]


# ------------------------------------------------------------------ session summary


def last_session(session: Session) -> dict | None:
    row = session.execute(
        select(AgentSessionRow).order_by(AgentSessionRow.started_at.desc()).limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    return {
        "id": row.id,
        "status": row.status,
        "metrics": row.metrics,
        "started_at": row.started_at.isoformat(),
        "stopped_at": row.stopped_at.isoformat() if row.stopped_at else None,
    }

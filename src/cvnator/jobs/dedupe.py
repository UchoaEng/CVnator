"""Duplicate and repost detection.

Checks run cheapest-first and stop at the first hit:

1. same platform + platform job id  - the same listing
2. same apply URL (normalised)      - the same listing reached two ways
3. same description hash            - a verbatim repost, possibly on another platform
4. same company/role/location key   - a repost with light edits
5. fuzzy description similarity     - a repost with a rewritten intro

Only an *applied* or *submitted* prior record blocks a new application. A job seen
but never applied to is not a duplicate, it is a retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from cvnator.db.models import ApplicationRow, JobRow
from cvnator.domain.application import ApplicationState
from cvnator.domain.job import JobPosting
from cvnator.observability import get_logger

log = get_logger(__name__)

REPOST_SIMILARITY_THRESHOLD = 92

_BLOCKING_STATES = {
    ApplicationState.SUBMITTED.value,
    ApplicationState.SUBMITTING.value,
    ApplicationState.REVIEW_PENDING.value,
}

#: Tracking parameters that differ between two links to the same posting.
_TRACKING_PREFIXES = ("utm_", "trk", "refid", "ref", "src", "source", "gh_src", "lever-source")


def normalize_url(url: str) -> str:
    """Strip tracking noise so two links to the same posting compare equal."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query = "&".join(
        piece
        for piece in parts.query.split("&")
        if piece and not piece.lower().startswith(_TRACKING_PREFIXES)
    )
    path = parts.path.rstrip("/")
    netloc = parts.netloc.lower().removeprefix("www.")
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def normalize_description(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


@dataclass(slots=True)
class DuplicateVerdict:
    is_duplicate: bool
    reason: str = ""
    existing_job_id: int | None = None
    is_repost: bool = False

    @classmethod
    def unique(cls) -> DuplicateVerdict:
        return cls(is_duplicate=False)


def _has_blocking_application(session: Session, job_id: int) -> bool:
    stmt = select(ApplicationRow.id).where(
        ApplicationRow.job_id == job_id,
        ApplicationRow.state.in_(_BLOCKING_STATES),
        ApplicationRow.dry_run.is_(False),
    )
    return session.execute(stmt).first() is not None


def find_duplicate(session: Session, job: JobPosting) -> DuplicateVerdict:
    """Decide whether ``job`` has effectively been applied to already."""
    if job.platform_job_id:
        row = session.execute(
            select(JobRow).where(
                JobRow.platform == job.platform,
                JobRow.platform_job_id == job.platform_job_id,
            )
        ).scalar_one_or_none()
        if row is not None and _has_blocking_application(session, row.id):
            return DuplicateVerdict(True, "Already applied via this platform listing", row.id)

    target_url = normalize_url(job.apply_url or job.url)
    if target_url:
        for row in session.execute(select(JobRow)).scalars():
            same_url = normalize_url(row.apply_url or row.url) == target_url
            if same_url and _has_blocking_application(session, row.id):
                return DuplicateVerdict(True, "Already applied via the same URL", row.id)

    description_hash = job.description_hash()
    if job.description.strip():
        row = session.execute(
            select(JobRow).where(JobRow.description_hash == description_hash)
        ).scalars().first()
        if row is not None and _has_blocking_application(session, row.id):
            return DuplicateVerdict(
                True, "Identical description already applied to", row.id, is_repost=True
            )

    identity = job.identity_key()
    candidates = list(
        session.execute(select(JobRow).where(JobRow.identity_key == identity)).scalars()
    )
    for row in candidates:
        if _has_blocking_application(session, row.id):
            return DuplicateVerdict(
                True,
                f"Already applied to {job.role} at {job.company} ({job.location})",
                row.id,
                is_repost=True,
            )

    normalized = normalize_description(job.description)
    if len(normalized) > 400:
        same_company = session.execute(
            select(JobRow).where(JobRow.company == job.company)
        ).scalars()
        for row in same_company:
            if not row.description:
                continue
            similarity = fuzz.ratio(normalized, normalize_description(row.description))
            if similarity >= REPOST_SIMILARITY_THRESHOLD and _has_blocking_application(
                session, row.id
            ):
                return DuplicateVerdict(
                    True,
                    f"Repost of an applied listing (similarity {similarity:.0f}%)",
                    row.id,
                    is_repost=True,
                )

    return DuplicateVerdict.unique()

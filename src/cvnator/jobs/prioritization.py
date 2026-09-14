"""Queue ordering for the application queue.

Deterministic on purpose: the user needs to be able to predict and explain why one
job was attempted before another, and the ranking is re-run whenever the queue
changes, so a model-produced ordering would be both expensive and unstable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from cvnator.domain.job import ApplyMethod, JobPosting
from cvnator.domain.profile import RemotePolicy
from cvnator.domain.search import SearchProfile

WEIGHTS = {
    "match": 0.45,
    "freshness": 0.12,
    "company_preference": 0.10,
    "salary": 0.10,
    "remote": 0.10,
    "simplicity": 0.08,
    "user_priority": 0.05,
}

#: Rough cost of completing each application route; cheaper routes go first.
_COMPLEXITY = {
    ApplyMethod.EASY_APPLY: 0.1,
    ApplyMethod.NATIVE_FORM: 0.4,
    ApplyMethod.EMAIL: 0.5,
    ApplyMethod.EXTERNAL: 0.8,
    ApplyMethod.UNKNOWN: 0.6,
}


@dataclass(slots=True)
class PrioritizedJob:
    job: JobPosting
    job_id: int
    match_score: float
    priority: float
    factors: dict[str, float]


def _freshness(posted_at: str) -> float:
    """1.0 for a posting from today, decaying to 0 over 30 days."""
    if not posted_at:
        return 0.5
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = datetime.strptime(posted_at[:26], fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - parsed).days
        return max(0.0, min(1.0, 1.0 - age_days / 30.0))
    return 0.5


def _salary_score(job: JobPosting, search: SearchProfile) -> float:
    if search.salary_minimum is None:
        return 0.5
    offered = job.salary.maximum or job.salary.minimum
    if offered is None:
        return 0.5
    ratio = offered / search.salary_minimum
    return max(0.0, min(1.0, (ratio - 0.8) / 0.7))


def _remote_score(job: JobPosting, search: SearchProfile) -> float:
    wanted = set(search.remote_policies)
    if not wanted or RemotePolicy.ANY in wanted:
        return 0.6
    if job.remote_policy in wanted:
        return 1.0
    if job.remote_policy is RemotePolicy.ANY:
        return 0.5
    return 0.2


def _company_preference(job: JobPosting, search: SearchProfile) -> float:
    company = job.company.strip().lower()
    if any(company == c.strip().lower() for c in search.companies_whitelist):
        return 1.0
    return 0.5


def score_priority(
    job: JobPosting, job_id: int, match_score: float, search: SearchProfile
) -> PrioritizedJob:
    factors = {
        "match": match_score,
        "freshness": _freshness(job.posted_at),
        "company_preference": _company_preference(job, search),
        "salary": _salary_score(job, search),
        "remote": _remote_score(job, search),
        "simplicity": 1.0 - _COMPLEXITY.get(job.apply_method, 0.6),
        "user_priority": max(0.0, min(1.0, search.priority / 10.0)),
    }
    priority = sum(WEIGHTS[name] * value for name, value in factors.items())
    return PrioritizedJob(
        job=job,
        job_id=job_id,
        match_score=match_score,
        priority=round(priority, 4),
        factors={k: round(v, 3) for k, v in factors.items()},
    )


def prioritize(items: list[PrioritizedJob]) -> list[PrioritizedJob]:
    """Highest priority first; ties broken by match score, then by newest."""
    return sorted(
        items,
        key=lambda i: (i.priority, i.match_score, _freshness(i.job.posted_at)),
        reverse=True,
    )

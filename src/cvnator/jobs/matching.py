"""Job matching: JOB x CANDIDATE MASTER PROFILE x SEARCH PROFILE -> score.

Two stages, in this order for a reason:

1. **Hard filters** (deterministic). Excluded keywords, blacklisted companies, missing
   required keywords, salary floor, remote policy. These are the user's explicit
   instructions, so a model never gets a vote on them - it would eventually talk
   itself into an exception.
2. **Scoring** (deterministic skill coverage + model judgement). Skill coverage is
   computed lexically so the number is reproducible; the model then judges the
   requirements a keyword match cannot settle ("experience leading a data team")
   and supplies the risks and advantages shown in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from cvnator.ai import ChatMessage, ModelRole
from cvnator.ai.gateway import AIGateway
from cvnator.domain.job import JobPosting
from cvnator.domain.profile import CandidateMasterProfile, RemotePolicy
from cvnator.domain.search import SearchProfile
from cvnator.observability import get_logger
from cvnator.resume.ats import ATSAnalysis
from cvnator.resume.vocabulary import expand_aliases, find_tech_terms

log = get_logger(__name__)

WEIGHT_SKILLS = 0.45
WEIGHT_REQUIREMENTS = 0.35
WEIGHT_TITLE = 0.12
WEIGHT_LANGUAGE = 0.08


@dataclass(slots=True)
class FilterResult:
    passed: bool
    reason: str = ""

    @classmethod
    def reject(cls, reason: str) -> FilterResult:
        return cls(passed=False, reason=reason)


class RequirementJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str
    satisfied: bool
    is_required: bool = True
    evidence: str = Field(default="", description="Where in the profile this is supported")


class MatchJudgement(BaseModel):
    """What the model contributes: requirement rulings, risks and advantages."""

    model_config = ConfigDict(extra="forbid")

    requirements: list[RequirementJudgement] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    advantages: list[str] = Field(default_factory=list)
    summary: str = ""


@dataclass(slots=True)
class MatchReport:
    overall: float
    skill_scores: dict[str, float] = field(default_factory=dict)
    required_satisfied: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    preferred_satisfied: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    advantages: list[str] = field(default_factory=list)
    summary: str = ""
    rejected_reason: str = ""

    @property
    def percent(self) -> int:
        return round(self.overall * 100)

    @property
    def rejected(self) -> bool:
        return bool(self.rejected_reason)

    def as_dict(self) -> dict[str, object]:
        return {
            "overall": round(self.overall, 4),
            "percent": self.percent,
            "skill_scores": {k: round(v, 3) for k, v in self.skill_scores.items()},
            "required_satisfied": self.required_satisfied,
            "missing_required": self.missing_required,
            "preferred_satisfied": self.preferred_satisfied,
            "risks": self.risks,
            "advantages": self.advantages,
            "summary": self.summary,
            "rejected_reason": self.rejected_reason,
        }


def apply_hard_filters(
    job: JobPosting, search: SearchProfile, profile: CandidateMasterProfile
) -> FilterResult:
    """The user's explicit rules. No model involvement, by design."""
    haystack = f"{job.role}\n{job.company}\n{job.description}"

    excluded = search.excludes(haystack)
    if excluded:
        return FilterResult.reject(f"Contains excluded keyword '{excluded}'")

    company = job.company.strip().lower()
    if any(company == c.strip().lower() for c in search.companies_blacklist):
        return FilterResult.reject(f"Company '{job.company}' is blacklisted")
    if search.companies_whitelist and not any(
        company == c.strip().lower() for c in search.companies_whitelist
    ):
        return FilterResult.reject(f"Company '{job.company}' is not on the whitelist")

    missing = search.missing_required_keywords(haystack)
    if missing:
        return FilterResult.reject(f"Missing required keywords: {', '.join(missing)}")

    policies = set(search.remote_policies)
    filtering_remote = bool(policies) and RemotePolicy.ANY not in policies
    # A posting that does not state its policy is given the benefit of the doubt.
    if filtering_remote and job.remote_policy not in policies | {RemotePolicy.ANY}:
        return FilterResult.reject(
            f"Remote policy '{job.remote_policy.value}' not in "
            f"{[p.value for p in policies]}"
        )

    if (
        search.seniorities
        and job.seniority.value != "unknown"
        and job.seniority not in search.seniorities
    ):
        return FilterResult.reject(f"Seniority '{job.seniority.value}' not wanted")

    if (
        search.salary_minimum is not None
        and job.salary.maximum is not None
        and job.salary.maximum < search.salary_minimum
    ):
        return FilterResult.reject(
            f"Salary ceiling {job.salary.maximum:g} below minimum "
            f"{search.salary_minimum:g}"
        )

    if profile.work_authorization.requires_sponsorship and job.visa_sponsorship is False:
        return FilterResult.reject("Job states no visa sponsorship and the profile requires it")

    return FilterResult(passed=True)


def score_skills(
    profile: CandidateMasterProfile, analysis: ATSAnalysis
) -> tuple[float, dict[str, float]]:
    """Per-technology coverage of what the posting asks for.

    A technology the profile lists explicitly scores 1.0; one that only appears in a
    role's free text scores 0.7; an alias match scores 0.9; anything else scores 0.
    """
    profile_explicit = {s.name.strip().lower() for s in profile.skills}
    profile_all = profile.skill_names()
    profile_corpus_terms = find_tech_terms(profile.as_fact_corpus())
    profile_aliased = expand_aliases(profile_all | profile_corpus_terms)

    wanted = [t.strip().lower() for t in analysis.technologies if t.strip()]
    if not wanted:
        return 1.0, {}

    scores: dict[str, float] = {}
    for tech in wanted:
        if tech in profile_explicit:
            scores[tech] = 1.0
        elif tech in profile_all or expand_aliases({tech}) & profile_aliased:
            scores[tech] = 0.9
        elif tech in profile_corpus_terms:
            scores[tech] = 0.7
        else:
            best = max(
                (fuzz.ratio(tech, known) for known in profile_aliased),
                default=0,
            )
            scores[tech] = 0.6 if best >= 90 else 0.0
    overall = sum(scores.values()) / len(scores)
    return overall, scores


def score_title(job: JobPosting, search: SearchProfile) -> float:
    titles = [search.role_title, *search.alternate_titles]
    role = job.role.strip().lower()
    if not role:
        return 0.5
    best = max((fuzz.token_set_ratio(role, t.strip().lower()) for t in titles if t), default=0)
    return min(best / 100.0, 1.0)


def score_language(job: JobPosting, profile: CandidateMasterProfile, analysis: ATSAnalysis) -> float:
    """Can the candidate operate in the posting's language?"""
    posting_language = (analysis.language or job.language or "").lower()[:2]
    if not posting_language:
        return 1.0
    aliases = {
        "pt": {"portugues", "português", "portuguese", "pt", "pt-br"},
        "en": {"ingles", "inglês", "english", "en"},
        "es": {"espanhol", "español", "spanish", "es", "castellano"},
    }.get(posting_language, set())
    if not aliases:
        return 0.8
    tiers = {"native": 1.0, "fluent": 1.0, "advanced": 0.9, "intermediate": 0.7, "basic": 0.4}
    for lang in profile.languages:
        if lang.language.strip().lower() in aliases:
            return tiers.get(lang.proficiency.value, 0.7)
    return 0.3


_JUDGE_SYSTEM = """\
You judge whether a candidate satisfies each requirement of a job posting.

Rules:
- Base every ruling ONLY on the candidate profile given. Never assume a skill from a
  job title, an industry, or a related technology.
- `satisfied` is true only when the profile contains concrete support. If you are
  unsure, answer false.
- `evidence` must quote or name the part of the profile that supports the ruling.
  If satisfied is false, leave evidence empty.
- `risks` are honest gaps a recruiter would notice. `advantages` are genuine
  strengths for this specific posting.
- Do not flatter the candidate and do not invent qualifications.
Return only JSON matching the schema.
"""


class JobMatcher:
    def __init__(self, gateway: AIGateway) -> None:
        self.gateway = gateway

    async def judge(
        self, profile: CandidateMasterProfile, job: JobPosting, analysis: ATSAnalysis
    ) -> MatchJudgement:
        requirements = analysis.requirements[:25]
        if not requirements:
            return MatchJudgement()
        requirement_lines = "\n".join(
            f"- ({'REQUIRED' if r.is_required else 'preferred'}) {r.text}" for r in requirements
        )
        messages = [
            ChatMessage(role="system", content=_JUDGE_SYSTEM),
            ChatMessage(
                role="user",
                content=(
                    f"CANDIDATE PROFILE:\n{profile.as_fact_corpus()[:8000]}\n\n"
                    f"TOTAL YEARS OF EXPERIENCE ON FILE: {profile.total_years_experience()}\n\n"
                    f"JOB: {job.role} at {job.company}\n\n"
                    f"REQUIREMENTS TO JUDGE:\n{requirement_lines}"
                ),
            ),
        ]
        try:
            return await self.gateway.structured(
                ModelRole.REASONING, MatchJudgement, messages, temperature=0.0, max_tokens=4096
            )
        except Exception as exc:
            log.warning("Match judgement failed for %s: %s", job.role, exc)
            return MatchJudgement(summary="Requirement judgement unavailable (model error)")

    async def match(
        self,
        profile: CandidateMasterProfile,
        job: JobPosting,
        search: SearchProfile,
        analysis: ATSAnalysis,
    ) -> MatchReport:
        hard = apply_hard_filters(job, search, profile)
        if not hard.passed:
            return MatchReport(overall=0.0, rejected_reason=hard.reason, summary=hard.reason)

        skill_overall, skill_scores = score_skills(profile, analysis)
        title_score = score_title(job, search)
        language_score = score_language(job, profile, analysis)
        judgement = await self.judge(profile, job, analysis)

        required = [r for r in judgement.requirements if r.is_required]
        preferred = [r for r in judgement.requirements if not r.is_required]
        required_satisfied = [r.requirement for r in required if r.satisfied]
        missing_required = [r.requirement for r in required if not r.satisfied]
        preferred_satisfied = [r.requirement for r in preferred if r.satisfied]

        requirement_score = (
            len(required_satisfied) / len(required) if required else (1.0 if not preferred else 0.8)
        )
        if preferred:
            requirement_score = 0.85 * requirement_score + 0.15 * (
                len(preferred_satisfied) / len(preferred)
            )

        overall = (
            WEIGHT_SKILLS * skill_overall
            + WEIGHT_REQUIREMENTS * requirement_score
            + WEIGHT_TITLE * title_score
            + WEIGHT_LANGUAGE * language_score
        )

        return MatchReport(
            overall=round(min(max(overall, 0.0), 1.0), 4),
            skill_scores=skill_scores,
            required_satisfied=required_satisfied,
            missing_required=missing_required,
            preferred_satisfied=preferred_satisfied,
            risks=judgement.risks,
            advantages=judgement.advantages,
            summary=judgement.summary,
        )

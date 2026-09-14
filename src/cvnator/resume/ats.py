"""ATS Intelligence.

Two things happen here:

* **Analysis** - turn a job description into the terms an applicant tracking system
  is likely to key on: technologies, hard skills, the role title and seniority,
  eliminating requirements, and the employer's own terminology.
* **Scoring** - estimate how well a given resume text covers that analysis.

An explicit disclaimer, because it matters: no ATS vendor publishes its ranking
algorithm, and this module does not claim to reproduce one. The number it produces
is CVnator's own internal coverage estimate, useful for comparing two versions of a
resume against the same posting. It is not a probability of being shortlisted, and
the UI must never present it as one.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from cvnator.ai import ChatMessage, ModelRole
from cvnator.ai.gateway import AIGateway
from cvnator.domain.job import JobPosting
from cvnator.observability import get_logger
from cvnator.resume.vocabulary import expand_aliases, find_tech_terms

log = get_logger(__name__)

#: How much each signal contributes to the coverage estimate.
WEIGHT_REQUIRED = 0.45
WEIGHT_TECHNOLOGIES = 0.30
WEIGHT_KEYWORDS = 0.15
WEIGHT_PREFERRED = 0.10

_ACTION_VERBS = {
    "develop", "build", "design", "implement", "maintain", "optimize", "analyze",
    "automate", "deliver", "lead", "manage", "coordinate", "monitor", "migrate",
    "integrate", "deploy", "support", "improve", "create", "define", "document",
    "desenvolver", "construir", "implementar", "manter", "otimizar", "analisar",
    "automatizar", "liderar", "gerenciar", "coordenar", "monitorar", "migrar",
    "integrar", "implantar", "apoiar", "melhorar", "criar", "definir", "documentar",
}

_STOP = {
    "the", "and", "for", "with", "you", "your", "our", "will", "have", "are", "that",
    "this", "from", "role", "team", "work", "who", "all", "not", "can", "job", "about",
    "de", "da", "do", "para", "com", "que", "uma", "dos", "das", "por", "sua", "seu",
    "como", "mais", "voce", "você", "nossa", "nosso", "ser", "sobre", "vaga", "empresa",
}


class ATSRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    is_required: bool = True
    is_eliminating: bool = False
    category: str = "skill"


class ATSAnalysis(BaseModel):
    """Structured reading of what a posting is screening for."""

    model_config = ConfigDict(extra="forbid")

    role_title: str = ""
    seniority: str = "unknown"
    language: str = ""
    technologies: list[str] = Field(default_factory=list)
    hard_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    action_verbs: list[str] = Field(default_factory=list)
    requirements: list[ATSRequirement] = Field(default_factory=list)
    employer_terminology: list[str] = Field(default_factory=list)

    @property
    def required_terms(self) -> list[str]:
        return [r.text for r in self.requirements if r.is_required]

    @property
    def preferred_terms(self) -> list[str]:
        return [r.text for r in self.requirements if not r.is_required]

    @property
    def eliminating_terms(self) -> list[str]:
        return [r.text for r in self.requirements if r.is_eliminating]


@dataclass(slots=True)
class ATSScore:
    """Internal compatibility estimate. Never a guarantee of anything."""

    match_score: float
    keywords_total: int
    keywords_matched: int
    requirements_total: int
    requirements_covered: int
    missing_requirements: list[str] = field(default_factory=list)
    missing_technologies: list[str] = field(default_factory=list)
    matched_technologies: list[str] = field(default_factory=list)

    @property
    def percent(self) -> int:
        return round(self.match_score * 100)

    def as_dict(self) -> dict[str, object]:
        return {
            "match_score": round(self.match_score, 4),
            "percent": self.percent,
            "keywords_total": self.keywords_total,
            "keywords_matched": self.keywords_matched,
            "requirements_total": self.requirements_total,
            "requirements_covered": self.requirements_covered,
            "missing_requirements": self.missing_requirements,
            "missing_technologies": self.missing_technologies,
            "matched_technologies": self.matched_technologies,
            "disclaimer": (
                "Internal coverage estimate produced by CVnator. Not a prediction of "
                "any employer's or ATS vendor's actual screening result."
            ),
        }


_ANALYZER_SYSTEM = """\
You analyse a job posting to identify what an applicant tracking system and a
recruiter will screen for. You extract; you do not invent.

Rules:
- Only list terms that actually appear in, or are directly stated by, the posting.
- `is_eliminating` is true only when the posting says the item is mandatory,
  a "must have", "requisito obrigatório", "eliminatório", or similar.
- `employer_terminology` captures the posting's own vocabulary where it differs from
  the industry-standard word (e.g. it says "Business Partner" for a generalist role).
- Keep every technology name in its original form. Never translate proper nouns.
- Detect `language` as the ISO code of the language the posting is written in
  ("pt", "en", "es").
Return only JSON matching the schema.
"""


def extract_keywords_lexically(text: str, limit: int = 40) -> list[str]:
    """Deterministic keyword pass: frequent, meaningful terms plus known technologies.

    Runs alongside the model so that the keyword set stays stable between runs even
    when the model's phrasing wobbles.
    """
    lowered = text.lower()
    tech = sorted(find_tech_terms(lowered))

    words = re.findall(r"[a-zà-ÿ][a-zà-ÿ0-9+#.\-]{2,}", lowered)
    counts = Counter(w for w in words if w not in _STOP and len(w) > 3)

    bigrams: Counter[str] = Counter()
    tokens = [w for w in words if w not in _STOP]
    for first, second in zip(tokens, tokens[1:], strict=False):
        if len(first) > 3 and len(second) > 3:
            bigrams[f"{first} {second}"] += 1

    ranked = [w for w, c in counts.most_common(limit) if c >= 2]
    ranked += [b for b, c in bigrams.most_common(limit // 2) if c >= 2]
    ordered = tech + [w for w in ranked if w not in tech]
    return ordered[:limit]


def extract_action_verbs(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({verb for verb in _ACTION_VERBS if re.search(rf"\b{verb}\w*\b", lowered)})


class ATSAnalyzer:
    def __init__(self, gateway: AIGateway) -> None:
        self.gateway = gateway

    async def analyze(self, job: JobPosting) -> ATSAnalysis:
        """Hybrid analysis: model for structure and intent, lexical pass for stability."""
        description = job.description or ""
        context = "\n".join(
            [
                f"Role: {job.role}",
                f"Company: {job.company}",
                f"Location: {job.location}",
                "",
                description,
            ]
        )
        messages = [
            ChatMessage(role="system", content=_ANALYZER_SYSTEM),
            ChatMessage(role="user", content=f"Job posting:\n\n---\n{context}\n---"),
        ]
        try:
            analysis = await self.gateway.structured(
                ModelRole.REASONING,
                ATSAnalysis,
                messages,
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception as exc:
            log.warning("ATS model analysis failed (%s); falling back to lexical only", exc)
            analysis = ATSAnalysis(role_title=job.role)

        lexical_keywords = extract_keywords_lexically(context)
        merged = list(dict.fromkeys([*analysis.keywords, *lexical_keywords]))
        analysis.keywords = merged

        # Merge the model's list with the lexical one, de-duplicating case-insensitively
        # so "AWS" and "aws" do not both count toward the score.
        lexical_tech = sorted(find_tech_terms(context))
        seen: set[str] = set()
        merged_tech: list[str] = []
        for term in [*analysis.technologies, *lexical_tech]:
            key = term.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged_tech.append(term.strip())
        analysis.technologies = merged_tech
        if not analysis.action_verbs:
            analysis.action_verbs = extract_action_verbs(context)
        if not analysis.role_title:
            analysis.role_title = job.role
        return analysis


def _covers(term: str, haystack: str, haystack_terms: set[str]) -> bool:
    """Is ``term`` represented in the resume text?

    Exact substring first; then alias-aware technology matching; then a fuzzy
    partial match for multi-word phrases that were reworded.
    """
    needle = term.strip().lower()
    if not needle:
        return False
    if needle in haystack:
        return True
    needle_terms = expand_aliases({needle}) if needle in find_tech_terms(needle) else set()
    if needle_terms & haystack_terms:
        return True
    if " " in needle and len(needle) > 8:
        return fuzz.partial_token_set_ratio(needle, haystack) >= 90
    return False


def score_resume(resume_text: str, analysis: ATSAnalysis) -> ATSScore:
    """Estimate how much of ``analysis`` the resume text covers."""
    haystack = resume_text.lower()
    haystack_terms = expand_aliases(find_tech_terms(haystack))

    required = analysis.required_terms
    preferred = analysis.preferred_terms
    technologies = [t.lower() for t in analysis.technologies]
    keywords = [k.lower() for k in analysis.keywords]

    def coverage(terms: list[str]) -> tuple[float, list[str], list[str]]:
        if not terms:
            return 1.0, [], []
        hit: list[str] = []
        miss: list[str] = []
        for term in terms:
            (hit if _covers(term, haystack, haystack_terms) else miss).append(term)
        return len(hit) / len(terms), hit, miss

    required_cov, _, missing_required = coverage(required)
    tech_cov, matched_tech, missing_tech = coverage(technologies)
    keyword_cov, matched_keywords, _ = coverage(keywords)
    preferred_cov, _, _ = coverage(preferred)

    score = (
        WEIGHT_REQUIRED * required_cov
        + WEIGHT_TECHNOLOGIES * tech_cov
        + WEIGHT_KEYWORDS * keyword_cov
        + WEIGHT_PREFERRED * preferred_cov
    )

    return ATSScore(
        match_score=round(min(max(score, 0.0), 1.0), 4),
        keywords_total=len(keywords),
        keywords_matched=len(matched_keywords),
        requirements_total=len(required) + len(preferred),
        requirements_covered=(len(required) - len(missing_required))
        + round(preferred_cov * len(preferred)),
        missing_requirements=missing_required,
        missing_technologies=missing_tech,
        matched_technologies=matched_tech,
    )

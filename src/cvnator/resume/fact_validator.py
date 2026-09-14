"""Fact Validator - the guard between what a model writes and what gets submitted.

Nothing generated (a resume bullet, a form answer, a cover note) reaches a real
application without passing through here. It runs three independent layers:

1. **Structural** - identity fields (company, title, dates, degree, certification)
   are never model output in the first place; the optimizer may only reorder and
   rewrite, and this layer asserts that invariant held.
2. **Lexical/numeric** - detects technology claims the master profile cannot support,
   and duration claims ("5 years of Python") larger than the profile justifies.
3. **Entailment** (optional, model-backed) - for rewritten prose that passes the
   first two layers but may still have drifted in meaning.

Layers 1 and 2 are deterministic, so a failure is always explainable and never
depends on a model behaving well. Layer 3 can only add violations, never clear them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from rapidfuzz import fuzz

from cvnator.domain.profile import CandidateMasterProfile
from cvnator.observability import get_logger
from cvnator.resume.vocabulary import expand_aliases, find_tech_terms

log = get_logger(__name__)

#: Fraction of the original bullet's content words that a rewrite must still carry.
#: Coverage, not similarity: expanding a bullet is legitimate, replacing it is not.
DEFAULT_COVERAGE_FLOOR = 0.60
#: Per-word fuzzy threshold when checking whether a content word survived the rewrite.
_WORD_MATCH_THRESHOLD = 82
#: Tolerance (years) when comparing a stated duration against the profile.
YEARS_TOLERANCE = 0.6

#: Function words carry no factual content, so they are excluded from coverage.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "and", "the", "for", "with", "from", "into", "that", "this", "these", "those",
        "was", "were", "been", "being", "are", "its", "our", "their", "his", "her",
        "using", "used", "use", "via", "per", "than", "then", "also", "such", "which",
        "who", "what", "when", "where", "how", "all", "any", "each", "more", "most",
        "other", "some", "only", "own", "same", "very", "can", "will", "just", "not",
        "e", "de", "da", "do", "das", "dos", "para", "com", "em", "no", "na", "nos",
        "nas", "um", "uma", "uns", "umas", "por", "que", "como", "ao", "aos", "se",
        "y", "en", "los", "las", "del", "al", "con", "una", "unos", "unas",
    }
)


def _content_words(text: str) -> list[str]:
    """Lowercase content words (length > 2, not a stopword, not pure punctuation)."""
    tokens = re.findall(r"[\w.+#/-]{3,}", text.lower())
    return [t.strip(".-/") for t in tokens if t.strip(".-/") and t not in _STOPWORDS]


def content_coverage(original: str, rewritten: str) -> float:
    """Fraction of the original's content words still represented in the rewrite.

    Each original word counts as preserved if it appears in the rewrite, or if some
    rewrite word is a close fuzzy match (covering inflection: "created" -> "creation",
    "dashboards" -> "dashboard").
    """
    source_words = _content_words(original)
    if not source_words:
        return 1.0
    target_words = _content_words(rewritten)
    if not target_words:
        return 0.0
    target_set = set(target_words)

    matched = 0
    for word in source_words:
        if word in target_set:
            matched += 1
            continue
        if any(fuzz.ratio(word, other) >= _WORD_MATCH_THRESHOLD for other in target_words):
            matched += 1
    return matched / len(source_words)


class ViolationKind(str, Enum):
    UNSUPPORTED_SKILL = "unsupported_skill"
    INFLATED_DURATION = "inflated_duration"
    UNGROUNDED_REWRITE = "ungrounded_rewrite"
    INVENTED_ENTITY = "invented_entity"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    UNSUPPORTED_AUTHORIZATION = "unsupported_authorization"


class Severity(str, Enum):
    BLOCK = "block"
    WARN = "warn"


@dataclass(slots=True)
class Violation:
    kind: ViolationKind
    severity: Severity
    message: str
    context: str = ""
    evidence: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class ValidationReport:
    violations: list[Violation] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.BLOCK]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def add(self, violation: Violation) -> None:
        self.violations.append(violation)

    def merge(self, other: ValidationReport) -> None:
        self.violations.extend(other.violations)
        self.checks_run.extend(other.checks_run)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks_run": self.checks_run,
            "violations": [v.as_dict() for v in self.violations],
        }

    def summary(self) -> str:
        if self.ok:
            return f"{len(self.checks_run)} checks passed"
        return "; ".join(f"{v.kind.value}: {v.message}" for v in self.blocking)


# "5 years of Python", "5+ anos de experiência com Python", "3 años de SQL"
_DURATION_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*\+?\s*"
    r"(?:years?|yrs?|anos?|años?)"
    r"(?:\s+(?:of|with|in|de|em|com|using|usando|con))?\s+"
    r"(?:experi[eê]ncia\s+(?:com|em|de)\s+)?"
    r"([A-Za-z0-9 .+#/\-]{2,40})",
    re.IGNORECASE,
)


class FactValidator:
    """Validates generated content against a Candidate Master Profile."""

    def __init__(
        self,
        profile: CandidateMasterProfile,
        *,
        coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
    ) -> None:
        self.profile = profile
        self.coverage_floor = coverage_floor
        self._supported_terms = expand_aliases(
            find_tech_terms(profile.as_fact_corpus()) | profile.skill_names()
        )
        self._companies = {e.company.strip().lower() for e in profile.experiences if e.company}
        self._titles = {e.job_title.strip().lower() for e in profile.experiences if e.job_title}
        self._institutions = {
            e.institution.strip().lower() for e in profile.education if e.institution
        }
        self._certifications = {
            c.name.strip().lower() for c in profile.certifications if c.name
        }
        self._languages = {lang.language.strip().lower() for lang in profile.languages}

    # ------------------------------------------------------------- layer 2 checks

    def check_skill_claims(self, text: str, context: str = "") -> list[Violation]:
        """Flag technology terms that the profile gives no basis for claiming.

        Aliases are expanded on the *supported* side only, so that a profile listing
        "kubernetes" covers a mention of "k8s" without the reverse producing two
        violations for a single mention.
        """
        mentioned = find_tech_terms(text)
        unsupported = sorted(
            term
            for term in mentioned
            if term not in self._supported_terms
            and not (expand_aliases({term}) & self._supported_terms)
        )
        return [
            Violation(
                kind=ViolationKind.UNSUPPORTED_SKILL,
                severity=Severity.BLOCK,
                message=f"'{term}' is not present in the candidate's master profile",
                context=context,
                evidence=term,
            )
            for term in unsupported
        ]

    def check_duration_claims(self, text: str, context: str = "") -> list[Violation]:
        """Flag "N years of X" claims larger than the profile supports."""
        violations: list[Violation] = []
        for match in _DURATION_PATTERN.finditer(text):
            try:
                claimed = float(match.group(1).replace(",", "."))
            except ValueError:  # pragma: no cover - regex guarantees numeric
                continue
            subject = match.group(2).strip().rstrip(".,;").lower()

            supported = self.profile.years_with_skill(subject)
            if supported is None:
                terms = find_tech_terms(subject)
                for term in terms:
                    candidate = self.profile.years_with_skill(term)
                    if candidate is not None:
                        supported = candidate
                        break
            if supported is None:
                # Generic phrasing such as "5 years of experience" -> compare to total.
                supported = self.profile.total_years_experience()

            if claimed > supported + YEARS_TOLERANCE:
                violations.append(
                    Violation(
                        kind=ViolationKind.INFLATED_DURATION,
                        severity=Severity.BLOCK,
                        message=(
                            f"Claimed {claimed:g} years of '{subject}' but the profile "
                            f"supports about {supported:g}"
                        ),
                        context=context,
                        evidence=match.group(0),
                    )
                )
        return violations

    def check_entities(self, text: str, context: str = "") -> list[Violation]:
        """Flag references to employers/schools/certifications not in the profile.

        Only fires on text that *looks* like an attribution ("at <Company>",
        "certified by <Issuer>"), because scanning every capitalised word produces
        noise, and identity fields are already structurally protected.
        """
        violations: list[Violation] = []
        pattern = re.compile(
            r"(?:\bat\s+|\bna\s+|\bno\s+|\bpara\s+a\s+|\bcertified\s+by\s+|\bcertificado\s+pela\s+)"
            r"([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})"
        )
        known = self._companies | self._institutions | self._certifications
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            lowered = name.lower()
            if lowered in known:
                continue
            if any(fuzz.token_set_ratio(lowered, entity) >= 88 for entity in known):
                continue
            violations.append(
                Violation(
                    kind=ViolationKind.INVENTED_ENTITY,
                    severity=Severity.WARN,
                    message=f"Reference to '{name}' does not match any employer or school on file",
                    context=context,
                    evidence=match.group(0),
                )
            )
        return violations

    def check_rewrite_grounding(
        self, original: str, rewritten: str, context: str = ""
    ) -> list[Violation]:
        """A rewrite must stay lexically anchored to its source bullet.

        Rewriting is allowed and expected ("built dashboards" -> "developed and
        maintained dashboards for operational KPI analysis"). Replacement is not.

        The coverage test is skipped when the rewrite is in a different language from
        the original, because comparing tokens across languages is meaningless. Those
        rewrites are routed to the entailment checker instead (see
        :meth:`needs_entailment`); the technology check below still applies, since
        technology names do not translate.
        """
        from cvnator.resume.entailment import same_language

        violations: list[Violation] = []
        if not original.strip():
            violations.append(
                Violation(
                    kind=ViolationKind.UNGROUNDED_REWRITE,
                    severity=Severity.BLOCK,
                    message="Rewritten content has no source bullet in the master profile",
                    context=context,
                    evidence=rewritten[:160],
                )
            )
            return violations

        cross_language = not same_language(original, rewritten)
        coverage = content_coverage(original, rewritten)
        if not cross_language and coverage < self.coverage_floor:
            violations.append(
                Violation(
                    kind=ViolationKind.UNGROUNDED_REWRITE,
                    severity=Severity.BLOCK,
                    message=(
                        f"Rewrite dropped too much of the original "
                        f"(coverage {coverage:.0%} < {self.coverage_floor:.0%})"
                    ),
                    context=context,
                    evidence=f"original={original[:110]!r} rewritten={rewritten[:110]!r}",
                )
            )

        # A rewrite may not introduce a technology that is not in the source bullet,
        # even if the candidate knows it - it would attribute the skill to that role.
        source_terms = expand_aliases(find_tech_terms(original))
        new_terms = {
            term for term in find_tech_terms(rewritten)
            if term not in source_terms and not (expand_aliases({term}) & source_terms)
        }
        for term in sorted(new_terms):
            violations.append(
                Violation(
                    kind=ViolationKind.UNSUPPORTED_CLAIM,
                    severity=Severity.BLOCK,
                    message=(
                        f"Rewrite introduces '{term}', which the original bullet "
                        "does not mention"
                    ),
                    context=context,
                    evidence=rewritten[:160],
                )
            )
        return violations

    def check_work_authorization(self, text: str, context: str = "") -> list[Violation]:
        """Work authorization may only be asserted from what the user stated."""
        auth = self.profile.work_authorization
        if auth.countries_authorized or auth.visa_status or auth.requires_sponsorship is not None:
            return []
        pattern = re.compile(
            r"\b(authoriz\w+ to work|work permit|visa|green card|eligible to work|"
            r"autoriza[çc][aã]o de trabalho|permiss[aã]o de trabalho)\b",
            re.IGNORECASE,
        )
        if pattern.search(text):
            return [
                Violation(
                    kind=ViolationKind.UNSUPPORTED_AUTHORIZATION,
                    severity=Severity.BLOCK,
                    message=(
                        "Work authorization is asserted but the profile records none. "
                        "Ask the user instead of answering."
                    ),
                    context=context,
                    evidence=text[:160],
                )
            ]
        return []

    # ------------------------------------------------------------------ entry points

    def validate_text(self, text: str, context: str = "") -> ValidationReport:
        """Full deterministic pass over a piece of free text (e.g. a form answer)."""
        report = ValidationReport(
            checks_run=["skill_claims", "duration_claims", "entities", "work_authorization"]
        )
        report.violations.extend(self.check_skill_claims(text, context))
        report.violations.extend(self.check_duration_claims(text, context))
        report.violations.extend(self.check_entities(text, context))
        report.violations.extend(self.check_work_authorization(text, context))
        return report

    def needs_entailment(self, original: str, rewritten: str) -> bool:
        """True when only a model can tell whether the rewrite stayed faithful.

        Two cases qualify: a translation (lexical overlap is meaningless) and a
        heavy but still-passing paraphrase near the coverage floor.
        """
        from cvnator.resume.entailment import same_language

        if not original.strip() or not rewritten.strip():
            return False
        if not same_language(original, rewritten):
            return True
        return content_coverage(original, rewritten) < (self.coverage_floor + 0.20)

    def validate_rewrite(
        self, original: str, rewritten: str, context: str = ""
    ) -> ValidationReport:
        report = ValidationReport(checks_run=["rewrite_grounding", "duration_claims"])
        report.violations.extend(self.check_rewrite_grounding(original, rewritten, context))
        report.violations.extend(self.check_duration_claims(rewritten, context))
        return report

"""Layer 3 of the fact validator: model-backed entailment for rewritten prose.

Layers 1 and 2 are lexical, and lexical checks break in one legitimate case: the
candidate's CV is in one language and the posting is in another. Translating a true
bullet is correct behaviour - the product requires answering in the form's language -
but a translated bullet shares almost no tokens with its source, so a coverage check
reads it as a fabrication.

This module handles that case, and only that case. It asks the model a narrow,
verifiable question: *does the rewrite assert anything the original does not?* The
answer can only ever **add** violations; it cannot clear one raised by a
deterministic layer. A model is not permitted to authorise a claim.

Language detection is a small function-word heuristic rather than a dependency: it
only has to separate pt/en/es well enough to know whether a lexical comparison is
meaningful.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from cvnator.ai import ChatMessage, ModelRole
from cvnator.ai.gateway import AIGateway
from cvnator.observability import get_logger
from cvnator.resume.fact_validator import Severity, Violation, ViolationKind

log = get_logger(__name__)

_LANGUAGE_MARKERS: dict[str, set[str]] = {
    "pt": {
        "de", "da", "do", "das", "dos", "para", "com", "em", "na", "no", "nas", "nos",
        "uma", "um", "que", "por", "como", "mais", "foi", "são", "ção", "ções", "e",
        "criação", "desenvolvimento", "dados", "gestão", "análise", "atendimento",
    },
    "es": {
        "de", "la", "el", "los", "las", "para", "con", "en", "una", "un", "que", "por",
        "como", "más", "fue", "son", "ción", "datos", "gestión", "análisis", "desarrollo",
    },
    "en": {
        "the", "of", "and", "for", "with", "in", "on", "to", "a", "an", "that", "by",
        "as", "was", "were", "are", "data", "management", "analysis", "development",
    },
}


def detect_language(text: str) -> str:
    """Rough pt/en/es detection. Returns "" when there is not enough signal."""
    words = re.findall(r"[a-zà-ÿ]+", (text or "").lower())
    if len(words) < 3:
        return ""
    counts = {
        lang: sum(1 for w in words if w in markers)
        for lang, markers in _LANGUAGE_MARKERS.items()
    }
    # Portuguese-specific orthography is a strong tiebreaker against Spanish.
    lowered = (text or "").lower()
    if re.search(r"[ãõ]|ção|ções|lh|nh", lowered):
        counts["pt"] += 3
    if re.search(r"[ñ]|ción|ciones", lowered):
        counts["es"] += 3

    best = max(counts, key=lambda k: counts[k])
    if counts[best] == 0:
        return ""
    ordered = sorted(counts.values(), reverse=True)
    if len(ordered) > 1 and ordered[0] == ordered[1]:
        return ""
    return best


def same_language(a: str, b: str) -> bool:
    """True when both texts look like the same language, or detection is inconclusive.

    Defaulting to True on an inconclusive result is the conservative choice: it keeps
    the strict lexical check in force rather than silently downgrading to entailment.
    """
    first, second = detect_language(a), detect_language(b)
    if not first or not second:
        return True
    return first == second


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%?")


def numbers_preserved(original: str, rewritten: str) -> tuple[bool, list[str]]:
    """Every figure in the original must survive the rewrite, and none may be added.

    Language-independent, so this stays in force for translations: "redução de 30%"
    may become "30% reduction", never "40% reduction" and never a new statistic.
    """
    def normalize(text: str) -> list[str]:
        return [m.group(0).replace(" ", "").replace(",", ".") for m in _NUMBER_RE.finditer(text)]

    source = normalize(original)
    target = normalize(rewritten)
    added = [n for n in target if n not in source]
    return (not added), added


class EntailmentVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported: bool = Field(
        description="True only if the rewrite asserts nothing the original does not"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims present in the rewrite but not the original"
    )
    is_faithful_translation: bool = Field(
        default=False, description="True if the rewrite is the original in another language"
    )
    reason: str = ""


_ENTAILMENT_SYSTEM = """\
You compare two statements about a person's work experience: an ORIGINAL taken from
their CV, and a REWRITE produced for a job application. The two may be in different
languages, which is allowed.

Answer one question: does the REWRITE assert anything that the ORIGINAL does not
support?

Rules:
- Rephrasing, expanding wording, and translating are all fine. Say supported=true.
- Adding a technology, tool, metric, number, scope, team size, seniority or outcome
  that the ORIGINAL does not contain is NOT fine. Say supported=false and list it.
- Making a vague statement more specific in a way the ORIGINAL does not justify is
  NOT fine.
- If you are unsure whether something was added, answer supported=false.
- Removing detail is acceptable; only additions matter.

Example (supported=true):
  ORIGINAL: "Criação de dashboards operacionais."
  REWRITE:  "Built and maintained dashboards for operational KPI monitoring."
  -> same fact, translated and slightly expanded.

Example (supported=false):
  ORIGINAL: "Criação de dashboards operacionais."
  REWRITE:  "Built Power BI dashboards that cut reporting time by 40%."
  -> adds a tool (Power BI) and a metric (40%) that the original does not contain.

Return only JSON matching the schema.
"""


class EntailmentChecker:
    """Model-backed check used only where lexical comparison cannot apply."""

    def __init__(self, gateway: AIGateway) -> None:
        self.gateway = gateway

    async def check(
        self, original: str, rewritten: str, context: str = ""
    ) -> list[Violation]:
        """Return violations for anything the rewrite adds. Never clears a violation."""
        ok, added_numbers = numbers_preserved(original, rewritten)
        violations: list[Violation] = []
        if not ok:
            violations.append(
                Violation(
                    kind=ViolationKind.UNSUPPORTED_CLAIM,
                    severity=Severity.BLOCK,
                    message=(
                        f"Rewrite introduces figures not in the original: "
                        f"{', '.join(added_numbers)}"
                    ),
                    context=context,
                    evidence=rewritten[:160],
                )
            )

        messages = [
            ChatMessage(role="system", content=_ENTAILMENT_SYSTEM),
            ChatMessage(
                role="user",
                content=f"ORIGINAL:\n{original}\n\nREWRITE:\n{rewritten}",
            ),
        ]
        try:
            verdict = await self.gateway.structured(
                ModelRole.REASONING,
                EntailmentVerdict,
                messages,
                temperature=0.0,
                max_tokens=768,
            )
        except Exception as exc:
            # A failed check must fail closed: without a verdict, the rewrite is not
            # cleared. The caller sees a blocking violation and falls back.
            log.warning("Entailment check failed (%s); treating the rewrite as unverified", exc)
            violations.append(
                Violation(
                    kind=ViolationKind.UNGROUNDED_REWRITE,
                    severity=Severity.BLOCK,
                    message=f"Rewrite could not be verified against the original: {exc}",
                    context=context,
                    evidence=rewritten[:160],
                )
            )
            return violations

        if not verdict.supported:
            claims = "; ".join(verdict.unsupported_claims) or verdict.reason
            violations.append(
                Violation(
                    kind=ViolationKind.UNSUPPORTED_CLAIM,
                    severity=Severity.BLOCK,
                    message=f"Rewrite asserts more than the original: {claims}"[:300],
                    context=context,
                    evidence=rewritten[:160],
                )
            )
        return violations

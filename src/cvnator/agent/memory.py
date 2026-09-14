"""Persistent agent memory.

The product promise is "configure once, run autonomously", and that only works if
the agent stops asking the same question. The first time a form asks for a desired
salary the agent does not know the answer, escalates, and records what the user
said. Every compatible posting afterwards is answered from memory.

Answers are **scoped**, because "desired salary" is not one fact:

    global < platform < country < role < company < job

A lookup returns the most specific answer that applies, so a company-specific salary
expectation beats the global one without overwriting it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from cvnator.db.models import MemoryRow
from cvnator.observability import get_logger

log = get_logger(__name__)


class MemoryCategory(str, Enum):
    CANDIDATE_FACT = "candidate_fact"
    USER_PREFERENCE = "user_preference"
    KNOWN_ANSWER = "known_answer"
    PLATFORM_KNOWLEDGE = "platform_knowledge"
    ACCOUNT_METADATA = "account_metadata"
    OPERATIONAL = "operational"


#: Higher wins when several scopes match the same question.
SCOPE_SPECIFICITY: dict[str, int] = {
    "global": 0,
    "platform": 1,
    "country": 2,
    "role": 3,
    "company": 4,
    "job": 5,
}

#: Below this, two question wordings are not treated as the same question.
QUESTION_MATCH_THRESHOLD = 86


def normalize_question(text: str) -> str:
    """A stable key for a question, robust to punctuation and phrasing noise."""
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"[*:?!.]+", " ", lowered)
    lowered = re.sub(r"\((required|obrigat[óo]rio|opcional|optional)\)", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered[:280]


@dataclass(slots=True)
class MemoryScope:
    """The situation a lookup is happening in."""

    platform: str = ""
    country: str = ""
    role: str = ""
    company: str = ""
    job_id: str = ""

    def as_pairs(self) -> list[tuple[str, str]]:
        pairs = [("global", "")]
        for name, value in (
            ("platform", self.platform),
            ("country", self.country),
            ("role", self.role),
            ("company", self.company),
            ("job", self.job_id),
        ):
            if value:
                pairs.append((name, value.strip().lower()))
        return pairs


@dataclass(slots=True)
class RecalledAnswer:
    value: str
    scope: str
    scope_value: str
    confidence: float
    question_text: str
    matched_by: str


class MemoryStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ---------------------------------------------------------------------- writing

    def remember(
        self,
        category: MemoryCategory,
        key: str,
        value: str,
        *,
        question_text: str = "",
        scope: str = "global",
        scope_value: str = "",
        confidence: float = 1.0,
        source: str = "user",
    ) -> MemoryRow:
        normalized_key = normalize_question(key) or key.strip().lower()
        scope_value = (scope_value or "").strip().lower()
        existing = self.session.execute(
            select(MemoryRow).where(
                MemoryRow.category == category.value,
                MemoryRow.key == normalized_key,
                MemoryRow.scope == scope,
                MemoryRow.scope_value == scope_value,
            )
        ).scalar_one_or_none()

        if existing is not None:
            existing.value = value
            existing.confidence = confidence
            existing.source = source
            existing.question_text = question_text or existing.question_text
            existing.updated_at = datetime.now(UTC)
            self.session.flush()
            return existing

        row = MemoryRow(
            category=category.value,
            key=normalized_key,
            question_text=question_text,
            value=value,
            scope=scope,
            scope_value=scope_value,
            confidence=confidence,
            source=source,
        )
        self.session.add(row)
        self.session.flush()
        log.info("Remembered %s '%s' (scope=%s%s)", category.value, normalized_key, scope,
                 f":{scope_value}" if scope_value else "")
        return row

    def remember_answer(
        self,
        question: str,
        answer: str,
        *,
        scope: str = "global",
        scope_value: str = "",
        source: str = "user",
    ) -> MemoryRow:
        return self.remember(
            MemoryCategory.KNOWN_ANSWER,
            question,
            answer,
            question_text=question,
            scope=scope,
            scope_value=scope_value,
            confidence=1.0 if source == "user" else 0.8,
            source=source,
        )

    # ---------------------------------------------------------------------- reading

    def recall_answer(
        self, question: str, scope: MemoryScope, *, min_confidence: float = 0.6
    ) -> RecalledAnswer | None:
        """Most specific applicable answer, exact key first then fuzzy question match."""
        key = normalize_question(question)
        applicable = scope.as_pairs()
        candidates: list[tuple[float, MemoryRow, str]] = []

        rows = list(
            self.session.execute(
                select(MemoryRow).where(
                    MemoryRow.category == MemoryCategory.KNOWN_ANSWER.value,
                    MemoryRow.confidence >= min_confidence,
                )
            ).scalars()
        )

        for row in rows:
            if (row.scope, row.scope_value) not in applicable:
                continue
            specificity = SCOPE_SPECIFICITY.get(row.scope, 0)
            if row.key == key:
                candidates.append((specificity * 10 + 5, row, "exact"))
                continue
            similarity = fuzz.token_set_ratio(key, row.key)
            if similarity >= QUESTION_MATCH_THRESHOLD:
                candidates.append((specificity * 10 + similarity / 100, row, "fuzzy"))

        if not candidates:
            return None
        _rank, row, matched_by = max(candidates, key=lambda item: item[0])
        row.hit_count += 1
        self.session.flush()
        return RecalledAnswer(
            value=row.value,
            scope=row.scope,
            scope_value=row.scope_value,
            confidence=row.confidence,
            question_text=row.question_text or row.key,
            matched_by=matched_by,
        )

    def recall(self, category: MemoryCategory, key: str) -> str | None:
        row = self.session.execute(
            select(MemoryRow).where(
                MemoryRow.category == category.value,
                MemoryRow.key == normalize_question(key),
            )
        ).scalars().first()
        return row.value if row else None

    def known_answers_for(self, scope: MemoryScope, limit: int = 40) -> dict[str, str]:
        """A digest of applicable answers, for priming the agent's context."""
        applicable = set(scope.as_pairs())
        rows = self.session.execute(
            select(MemoryRow).where(
                MemoryRow.category == MemoryCategory.KNOWN_ANSWER.value
            ).order_by(MemoryRow.hit_count.desc())
        ).scalars()
        result: dict[str, str] = {}
        for row in rows:
            if (row.scope, row.scope_value) in applicable and row.key not in result:
                result[row.question_text or row.key] = row.value
            if len(result) >= limit:
                break
        return result

    def platform_note(self, platform: str, key: str, value: str) -> None:
        """Record something learned about how a platform behaves."""
        self.remember(
            MemoryCategory.PLATFORM_KNOWLEDGE,
            f"{platform}:{key}",
            value,
            scope="platform",
            scope_value=platform.lower(),
            source="agent",
            confidence=0.7,
        )

    def forget(self, row_id: int) -> None:
        row = self.session.get(MemoryRow, row_id)
        if row is not None:
            self.session.delete(row)
            self.session.flush()

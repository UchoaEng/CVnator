"""Search profiles - the user's "what I am looking for" configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cvnator.domain.job import Seniority
from cvnator.domain.profile import RemotePolicy


class SearchProfile(BaseModel):
    """One saved search. A user may keep several, enabled independently."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    name: str
    enabled: bool = True

    role_title: str
    alternate_titles: list[str] = Field(default_factory=list)
    role_description: str = ""

    keywords: list[str] = Field(default_factory=list)
    required_keywords: list[str] = Field(default_factory=list)
    excluded_keywords: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)

    location: str = ""
    remote_policies: list[RemotePolicy] = Field(default_factory=lambda: [RemotePolicy.ANY])
    seniorities: list[Seniority] = Field(default_factory=list)

    salary_currency: str = ""
    salary_minimum: float | None = None

    languages: list[str] = Field(default_factory=list)
    companies_whitelist: list[str] = Field(default_factory=list)
    companies_blacklist: list[str] = Field(default_factory=list)

    platforms: list[str] = Field(default_factory=lambda: ["mock"])
    minimum_match_score: float = 0.70
    daily_application_limit: int = 10
    priority: int = 0

    def excludes(self, text: str) -> str | None:
        """Return the first excluded keyword present in ``text``, if any."""
        haystack = text.lower()
        for word in self.excluded_keywords:
            token = word.strip().lower()
            if token and token in haystack:
                return word
        return None

    def missing_required_keywords(self, text: str) -> list[str]:
        haystack = text.lower()
        return [w for w in self.required_keywords if w.strip() and w.strip().lower() not in haystack]


class AutonomySettings(BaseModel):
    """Per-session autonomy switches. Enforced by the policy engine, not by the LLM."""

    model_config = ConfigDict(extra="forbid")

    auto_search: bool = True
    auto_match: bool = True
    auto_optimize_resume: bool = True
    auto_apply: bool = True
    auto_submit: bool = False
    auto_create_external_account: bool = False
    account_policy: str = "ask"  # never | ask | auto
    delete_temp_resumes: bool = True
    dry_run: bool = True
    review_before_submit: bool = True
    minimum_match_score: float = 0.70
    daily_application_limit: int = 20
    session_application_limit: int = 50

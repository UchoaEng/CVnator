"""Normalized job posting - the shape every platform adapter must produce."""

from __future__ import annotations

import hashlib
import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from cvnator.domain.profile import RemotePolicy


class Seniority(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    MANAGER = "manager"
    UNKNOWN = "unknown"


class ApplyMethod(str, Enum):
    EASY_APPLY = "easy_apply"
    EXTERNAL = "external"
    NATIVE_FORM = "native_form"
    EMAIL = "email"
    UNKNOWN = "unknown"


class SalaryRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: str = ""
    minimum: float | None = None
    maximum: float | None = None
    period: str = ""


class JobRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    is_required: bool = True
    category: str = "general"  # skill | experience | education | language | authorization


class JobPosting(BaseModel):
    """Structured representation of a posting, produced by the Job Intelligence step."""

    model_config = ConfigDict(extra="forbid")

    platform: str = "unknown"
    platform_job_id: str = ""
    external_job_id: str = ""
    url: str = ""
    apply_url: str = ""

    company: str = ""
    role: str = ""
    location: str = ""
    remote_policy: RemotePolicy = RemotePolicy.ANY
    seniority: Seniority = Seniority.UNKNOWN
    language: str = ""
    salary: SalaryRange = Field(default_factory=SalaryRange)

    description: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[JobRequirement] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    years_of_experience: float | None = None
    visa_sponsorship: bool | None = None
    work_authorization_note: str = ""

    apply_method: ApplyMethod = ApplyMethod.UNKNOWN
    application_platform: str = ""
    posted_at: str = ""

    @property
    def required_requirements(self) -> list[JobRequirement]:
        return [r for r in self.requirements if r.is_required]

    @property
    def preferred_requirements(self) -> list[JobRequirement]:
        return [r for r in self.requirements if not r.is_required]

    def description_hash(self) -> str:
        """Stable hash of the normalized description, used for repost detection."""
        normalized = re.sub(r"\s+", " ", self.description or "").strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def identity_key(self) -> str:
        """Platform-independent identity used for duplicate detection."""
        parts = [
            (self.company or "").strip().lower(),
            (self.role or "").strip().lower(),
            re.sub(r"[^a-z0-9]+", "", (self.location or "").strip().lower()),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

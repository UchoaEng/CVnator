"""Candidate Master Profile - the single source of truth about the candidate.

Every generated resume, every form answer and every claim the agent makes must be
traceable back to this structure. ``cvnator.resume.fact_validator`` enforces that.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from cvnator.domain.dates import months_between


class Proficiency(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    FLUENT = "fluent"
    NATIVE = "native"


class RemotePolicy(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    ANY = "any"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PersonalInformation(StrictModel):
    full_name: str = ""
    email: str = ""
    phone: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""
    date_of_birth: date | None = None


class Experience(StrictModel):
    company: str
    job_title: str
    # Free-form "YYYY-MM" strings: CV input is messy and normalising too early loses data.
    start_date: str = ""
    end_date: str = ""
    is_current: bool = False
    location: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class Project(StrictModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    url: str = ""
    highlights: list[str] = Field(default_factory=list)


class Education(StrictModel):
    institution: str
    degree: str = ""
    field_of_study: str = ""
    start_date: str = ""
    end_date: str = ""
    status: str = ""


class Certification(StrictModel):
    name: str
    issuer: str = ""
    issue_date: str = ""
    expiry_date: str = ""
    credential_id: str = ""


class LanguageSkill(StrictModel):
    language: str
    proficiency: Proficiency = Proficiency.INTERMEDIATE


class Skill(StrictModel):
    name: str
    category: str = "hard"
    years_of_experience: float | None = None
    proficiency: Proficiency | None = None
    evidence: list[str] = Field(
        default_factory=list,
        description="Companies/projects where this skill was used; read by the fact validator.",
    )


class SalaryPreferences(StrictModel):
    currency: str = "BRL"
    minimum: float | None = None
    target: float | None = None
    period: str = "month"
    negotiable: bool = True


class WorkPreferences(StrictModel):
    remote_policy: RemotePolicy = RemotePolicy.ANY
    willing_to_relocate: bool = False
    preferred_locations: list[str] = Field(default_factory=list)
    notice_period_days: int | None = None
    available_from: str = ""
    contract_types: list[str] = Field(default_factory=list)


class WorkAuthorization(StrictModel):
    """Only ever populated from what the user states. Never inferred by a model."""

    countries_authorized: list[str] = Field(default_factory=list)
    requires_sponsorship: bool | None = None
    visa_status: str = ""
    notes: str = ""


class AdditionalAnswer(StrictModel):
    """A stored answer to a recurring application question."""

    question_key: str
    question_text: str = ""
    answer: str
    scope: str = "global"
    scope_value: str = ""


class CandidateMasterProfile(StrictModel):
    """Serializable source of truth, persisted as JSON alongside database rows."""

    version: int = 1
    personal_information: PersonalInformation = Field(default_factory=PersonalInformation)
    professional_summary: str = ""
    headline: str = ""
    experiences: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[LanguageSkill] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    salary_preferences: SalaryPreferences = Field(default_factory=SalaryPreferences)
    work_preferences: WorkPreferences = Field(default_factory=WorkPreferences)
    work_authorization: WorkAuthorization = Field(default_factory=WorkAuthorization)
    additional_answers: list[AdditionalAnswer] = Field(default_factory=list)
    master_cv_path: str = ""
    master_cv_text: str = ""

    @property
    def hard_skills(self) -> list[Skill]:
        return [s for s in self.skills if s.category == "hard"]

    @property
    def soft_skills(self) -> list[Skill]:
        return [s for s in self.skills if s.category == "soft"]

    def skill_names(self) -> set[str]:
        """Every skill or technology the profile can vouch for, lowercased."""
        names: set[str] = {s.name.strip().lower() for s in self.skills if s.name.strip()}
        for exp in self.experiences:
            names.update(t.strip().lower() for t in exp.technologies if t.strip())
        for proj in self.projects:
            names.update(t.strip().lower() for t in proj.technologies if t.strip())
        names.discard("")
        return names

    def total_years_experience(self) -> float:
        """Rough sum of experience spans. A signal for ranking, never a claim to submit."""
        months = sum(months_between(e.start_date, e.end_date or None) for e in self.experiences)
        return round(months / 12.0, 1)

    def years_with_skill(self, skill_name: str) -> float | None:
        """Years with a named skill, or ``None`` when the profile gives no basis for a number.

        Returning ``None`` is meaningful: it makes the agent ask the user instead of
        inventing a duration.
        """
        target = skill_name.strip().lower()
        for skill in self.skills:
            if skill.name.strip().lower() == target and skill.years_of_experience is not None:
                return skill.years_of_experience
        months = 0
        matched = False
        for exp in self.experiences:
            if target in {t.strip().lower() for t in exp.technologies}:
                matched = True
                months += months_between(exp.start_date, exp.end_date or None)
        return round(months / 12.0, 1) if matched else None

    def as_fact_corpus(self) -> str:
        """Flattened text of every verifiable fact. Fed to the fact validator."""
        parts: list[str] = [self.professional_summary, self.headline]
        for exp in self.experiences:
            parts += [exp.company, exp.job_title, exp.location]
            parts += exp.responsibilities + exp.achievements + exp.technologies
        for proj in self.projects:
            parts += [proj.name, proj.description]
            parts += proj.highlights + proj.technologies
        for edu in self.education:
            parts += [edu.institution, edu.degree, edu.field_of_study]
        for cert in self.certifications:
            parts += [cert.name, cert.issuer]
        parts += [lang.language for lang in self.languages]
        parts += [skill.name for skill in self.skills]
        return "\n".join(p for p in parts if p)

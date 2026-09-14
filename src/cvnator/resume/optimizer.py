"""Resume Optimizer.

The central safety idea: **the model is never allowed to write identity data.**

It receives the master profile with every bullet numbered, and it may only answer
with *references* - "keep experience 2 before experience 0", "rewrite bullet 1 of
experience 2 as this text", "put SQL before Python in the skills list". Company
names, job titles, employment dates, degrees, certifications and languages are then
copied verbatim from the master profile by :func:`build_document`.

That makes an invented employer structurally impossible rather than merely
discouraged, and narrows what the fact validator has to police down to the prose it
actually generated: the headline, the summary, and the rewritten bullets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from cvnator.ai import ChatMessage, ModelRole
from cvnator.ai.gateway import AIGateway
from cvnator.domain.job import JobPosting
from cvnator.domain.profile import CandidateMasterProfile
from cvnator.observability import get_logger
from cvnator.resume.ats import ATSAnalysis
from cvnator.resume.fact_validator import (
    FactValidator,
    Severity,
    ValidationReport,
    Violation,
    ViolationKind,
)

log = get_logger(__name__)


class BulletRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_kind: str = Field(description="'responsibility' or 'achievement'")
    source_index: int = Field(description="Index of the original bullet within that list")
    text: str = Field(description="Rewritten wording; must state the same fact")


class ExperienceRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experience_index: int
    include: bool = True
    bullets: list[BulletRewrite] = Field(default_factory=list)


class OptimizationPlan(BaseModel):
    """What the model is permitted to decide. Everything else is copied, not generated."""

    model_config = ConfigDict(extra="forbid")

    headline: str = ""
    professional_summary: str = ""
    experience_order: list[int] = Field(default_factory=list)
    experiences: list[ExperienceRewrite] = Field(default_factory=list)
    skill_order: list[str] = Field(default_factory=list)
    highlighted_skills: list[str] = Field(default_factory=list)
    project_order: list[int] = Field(default_factory=list)
    reasoning: str = ""


@dataclass(slots=True)
class RenderedExperience:
    company: str
    job_title: str
    start_date: str
    end_date: str
    is_current: bool
    location: str
    bullets: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RenderedProject:
    name: str
    description: str
    technologies: list[str] = field(default_factory=list)
    url: str = ""


@dataclass(slots=True)
class ResumeDocument:
    """Render-ready resume. Identity fields here came from the profile, not the model."""

    full_name: str
    contact_lines: list[str]
    headline: str
    professional_summary: str
    experiences: list[RenderedExperience]
    skills: list[str]
    projects: list[RenderedProject]
    education: list[str]
    certifications: list[str]
    languages: list[str]

    def as_text(self) -> str:
        """Plain-text projection, used for ATS scoring and for diffing versions."""
        parts: list[str] = [self.full_name, " | ".join(self.contact_lines)]
        if self.headline:
            parts.append(self.headline)
        if self.professional_summary:
            parts += ["", "PROFESSIONAL SUMMARY", self.professional_summary]
        if self.experiences:
            parts += ["", "EXPERIENCE"]
            for exp in self.experiences:
                period = f"{exp.start_date} - {'Present' if exp.is_current else exp.end_date}"
                parts.append(f"{exp.job_title} | {exp.company} | {exp.location} | {period}")
                parts += [f"- {b}" for b in exp.bullets]
                if exp.technologies:
                    parts.append("Technologies: " + ", ".join(exp.technologies))
        if self.skills:
            parts += ["", "SKILLS", ", ".join(self.skills)]
        if self.projects:
            parts += ["", "PROJECTS"]
            for proj in self.projects:
                parts.append(f"{proj.name}: {proj.description}")
                if proj.technologies:
                    parts.append("Technologies: " + ", ".join(proj.technologies))
        if self.education:
            parts += ["", "EDUCATION", *self.education]
        if self.certifications:
            parts += ["", "CERTIFICATIONS", *self.certifications]
        if self.languages:
            parts += ["", "LANGUAGES", ", ".join(self.languages)]
        return "\n".join(parts)

    def generated_prose(self) -> list[tuple[str, str]]:
        """Every piece of model-written text, paired with a context label."""
        items = [("headline", self.headline), ("summary", self.professional_summary)]
        for exp in self.experiences:
            items += [(f"{exp.company} bullet", b) for b in exp.bullets]
        return [(label, text) for label, text in items if text.strip()]


_OPTIMIZER_SYSTEM = """\
You tailor an existing resume to a specific job posting. You are an editor with
strictly limited powers.

YOU MAY:
- Write a headline and a professional summary, built only from facts already in the
  profile below.
- Reorder experiences, skills and projects so the most relevant come first.
- Rewrite an existing bullet point to use the posting's terminology and to read more
  precisely, as long as it states THE SAME FACT as the original.
- Choose which experiences to include (set include=false to drop an irrelevant one).

YOU MAY NOT:
- Invent or alter a company, job title, date, degree, certification or language.
  You are not given a way to write those fields; do not try.
- Add a technology, tool, metric, result or responsibility that the original bullet
  does not already contain. If the posting wants Kubernetes and the candidate has
  never used it, the correct behaviour is to leave it out.
- Inflate durations or seniority.
- Reference a bullet index that does not exist.

REWRITE EXAMPLE (this is the level of change that is acceptable):
  original:  "Criação de dashboards operacionais."
  rewritten: "Desenvolvimento e acompanhamento de dashboards para análise de
              indicadores operacionais e monitoramento de performance."
It is more specific, uses the posting's vocabulary, and adds no new fact.

Write in the language of the job posting. Keep proper nouns and technology names in
their original form. Return only JSON matching the schema.
"""


def _profile_briefing(profile: CandidateMasterProfile) -> str:
    """The profile, with every rewritable bullet numbered so the model can reference it."""
    lines: list[str] = []
    if profile.professional_summary:
        lines += ["CURRENT SUMMARY:", profile.professional_summary, ""]
    lines.append("EXPERIENCES:")
    for idx, exp in enumerate(profile.experiences):
        period = f"{exp.start_date} to {'present' if exp.is_current else exp.end_date}"
        lines.append(f"[experience {idx}] {exp.job_title} at {exp.company} ({period})")
        for bullet_idx, bullet in enumerate(exp.responsibilities):
            lines.append(f"    responsibility {bullet_idx}: {bullet}")
        for bullet_idx, bullet in enumerate(exp.achievements):
            lines.append(f"    achievement {bullet_idx}: {bullet}")
        if exp.technologies:
            lines.append(f"    technologies: {', '.join(exp.technologies)}")
    if profile.projects:
        lines.append("")
        lines.append("PROJECTS:")
        for idx, proj in enumerate(profile.projects):
            techs = f" [{', '.join(proj.technologies)}]" if proj.technologies else ""
            lines.append(f"[project {idx}] {proj.name}: {proj.description}{techs}")
    lines.append("")
    lines.append("SKILLS (use these exact names): " + ", ".join(s.name for s in profile.skills))
    if profile.languages:
        lines.append(
            "LANGUAGES: " + ", ".join(f"{lang.language} ({lang.proficiency.value})"
                                      for lang in profile.languages)
        )
    return "\n".join(lines)


def _job_briefing(job: JobPosting, analysis: ATSAnalysis) -> str:
    lines = [
        f"ROLE: {job.role}",
        f"COMPANY: {job.company}",
        f"LANGUAGE OF POSTING: {analysis.language or job.language or 'unknown'}",
        f"TECHNOLOGIES SOUGHT: {', '.join(analysis.technologies[:30])}",
        f"KEY TERMS: {', '.join(analysis.keywords[:30])}",
        f"EMPLOYER TERMINOLOGY: {', '.join(analysis.employer_terminology[:15])}",
        "",
        "REQUIREMENTS:",
    ]
    for req in analysis.requirements[:30]:
        mark = "REQUIRED" if req.is_required else "preferred"
        lines.append(f"  - ({mark}) {req.text}")
    lines += ["", "DESCRIPTION:", job.description[:6000]]
    return "\n".join(lines)


def build_document(
    profile: CandidateMasterProfile, plan: OptimizationPlan
) -> tuple[ResumeDocument, list[Violation]]:
    """Merge a plan with the master profile. Identity fields are copied, never generated.

    Any part of the plan that references something that does not exist is dropped and
    reported as a violation rather than silently ignored.
    """
    violations: list[Violation] = []
    rewrites_by_index = {r.experience_index: r for r in plan.experiences}

    order = [i for i in plan.experience_order if 0 <= i < len(profile.experiences)]
    for bad in set(plan.experience_order) - set(order):
        violations.append(
            Violation(
                kind=ViolationKind.INVENTED_ENTITY,
                severity=Severity.WARN,
                message=f"Plan referenced experience index {bad}, which does not exist",
                context="experience_order",
            )
        )
    order += [i for i in range(len(profile.experiences)) if i not in order]

    rendered: list[RenderedExperience] = []
    for idx in order:
        source = profile.experiences[idx]
        rewrite = rewrites_by_index.get(idx)
        if rewrite is not None and not rewrite.include:
            continue

        bullets: list[str] = []
        if rewrite is not None and rewrite.bullets:
            for item in rewrite.bullets:
                pool = (
                    source.achievements
                    if item.source_kind.startswith("achie")
                    else source.responsibilities
                )
                if not (0 <= item.source_index < len(pool)):
                    violations.append(
                        Violation(
                            kind=ViolationKind.UNGROUNDED_REWRITE,
                            severity=Severity.BLOCK,
                            message=(
                                f"Rewrite references {item.source_kind} "
                                f"{item.source_index} of experience {idx}, which does not exist"
                            ),
                            context=f"{source.company}",
                            evidence=item.text[:160],
                        )
                    )
                    continue
                bullets.append(item.text.strip())
        if not bullets:
            bullets = [*source.responsibilities, *source.achievements]

        rendered.append(
            RenderedExperience(
                company=source.company,
                job_title=source.job_title,
                start_date=source.start_date,
                end_date=source.end_date,
                is_current=source.is_current,
                location=source.location,
                bullets=bullets,
                technologies=source.technologies,
            )
        )

    known_skills = {s.name.strip().lower(): s.name for s in profile.skills}
    ordered_skills: list[str] = []
    for name in plan.skill_order:
        canonical = known_skills.get(name.strip().lower())
        if canonical is None:
            violations.append(
                Violation(
                    kind=ViolationKind.UNSUPPORTED_SKILL,
                    severity=Severity.BLOCK,
                    message=f"Plan lists skill '{name}', which is not in the master profile",
                    context="skill_order",
                    evidence=name,
                )
            )
            continue
        if canonical not in ordered_skills:
            ordered_skills.append(canonical)
    ordered_skills += [s.name for s in profile.skills if s.name not in ordered_skills]

    project_order = [i for i in plan.project_order if 0 <= i < len(profile.projects)]
    project_order += [i for i in range(len(profile.projects)) if i not in project_order]
    projects = [
        RenderedProject(
            name=profile.projects[i].name,
            description=profile.projects[i].description,
            technologies=profile.projects[i].technologies,
            url=profile.projects[i].url,
        )
        for i in project_order
    ]

    info = profile.personal_information
    contact = [
        c
        for c in (
            info.email,
            info.phone,
            ", ".join(p for p in (info.city, info.state, info.country) if p),
            info.linkedin,
            info.github,
            info.portfolio,
        )
        if c
    ]

    document = ResumeDocument(
        full_name=info.full_name,
        contact_lines=contact,
        headline=plan.headline.strip() or profile.headline,
        professional_summary=plan.professional_summary.strip() or profile.professional_summary,
        experiences=rendered,
        skills=ordered_skills,
        projects=projects,
        education=[
            " | ".join(
                p
                for p in (
                    e.degree,
                    e.field_of_study,
                    e.institution,
                    f"{e.start_date} - {e.end_date}".strip(" -"),
                    e.status,
                )
                if p
            )
            for e in profile.education
        ],
        certifications=[
            " | ".join(p for p in (c.name, c.issuer, c.issue_date) if p)
            for c in profile.certifications
        ],
        languages=[f"{lang.language} ({lang.proficiency.value})" for lang in profile.languages],
    )
    return document, violations


class ResumeOptimizer:
    def __init__(self, gateway: AIGateway) -> None:
        self.gateway = gateway

    async def plan(
        self, profile: CandidateMasterProfile, job: JobPosting, analysis: ATSAnalysis
    ) -> OptimizationPlan:
        messages = [
            ChatMessage(role="system", content=_OPTIMIZER_SYSTEM),
            ChatMessage(
                role="user",
                content=(
                    "CANDIDATE PROFILE (the only facts you may use):\n"
                    f"{_profile_briefing(profile)}\n\n"
                    "TARGET JOB:\n"
                    f"{_job_briefing(job, analysis)}"
                ),
            ),
        ]
        return await self.gateway.structured(
            ModelRole.RESUME,
            OptimizationPlan,
            messages,
            temperature=0.2,
            max_tokens=6144,
        )

    async def validate(
        self,
        profile: CandidateMasterProfile,
        plan: OptimizationPlan,
        document: ResumeDocument,
        structural_violations: list[Violation],
    ) -> ValidationReport:
        """Run the fact validator over everything the model actually wrote.

        Deterministic checks first. Where they cannot decide - a bullet translated
        into the posting's language, or a heavy paraphrase - the entailment checker
        is asked as well. It can only add violations, never clear one.
        """
        from cvnator.resume.entailment import EntailmentChecker

        validator = FactValidator(profile)
        entailment = EntailmentChecker(self.gateway)
        report = ValidationReport(checks_run=["structural", "prose", "bullet_grounding"])
        report.violations.extend(structural_violations)

        for label, text in (
            ("headline", document.headline),
            ("professional_summary", document.professional_summary),
        ):
            if text.strip():
                report.merge(validator.validate_text(text, context=label))

        rewrites_by_index = {r.experience_index: r for r in plan.experiences}
        for idx, rewrite in rewrites_by_index.items():
            if not (0 <= idx < len(profile.experiences)):
                continue
            source = profile.experiences[idx]
            for item in rewrite.bullets:
                pool = (
                    source.achievements
                    if item.source_kind.startswith("achie")
                    else source.responsibilities
                )
                original = pool[item.source_index] if 0 <= item.source_index < len(pool) else ""
                label = f"{source.company} / {source.job_title}"
                report.merge(validator.validate_rewrite(original, item.text, context=label))
                if original and validator.needs_entailment(original, item.text):
                    if "entailment" not in report.checks_run:
                        report.checks_run.append("entailment")
                    report.violations.extend(
                        await entailment.check(original, item.text, context=label)
                    )
        return report

    async def optimize(
        self, profile: CandidateMasterProfile, job: JobPosting, analysis: ATSAnalysis
    ) -> tuple[ResumeDocument, OptimizationPlan, ValidationReport]:
        """Produce a tailored resume, or report exactly why it could not be trusted."""
        plan = await self.plan(profile, job, analysis)
        document, structural = build_document(profile, plan)
        report = await self.validate(profile, plan, document, structural)

        if not report.ok:
            log.warning(
                "Optimization plan for %s at %s failed validation: %s",
                job.role,
                job.company,
                report.summary(),
            )
        return document, plan, report


def baseline_document(profile: CandidateMasterProfile) -> ResumeDocument:
    """The untouched master profile rendered as a resume, for before/after comparison."""
    document, _ = build_document(profile, OptimizationPlan())
    return document

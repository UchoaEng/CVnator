"""Gupy adapter, plus the Gupy Profile Optimizer.

**Status: scaffold, not validated against the live site.** The navigation below is
written from Gupy's public structure and has not been run against production.

Gupy differs from a PDF-upload ATS in a way that matters for this product: the
recruiter reads the *structured profile stored on Gupy*, not only the attached file.
Swapping in a tailored PDF therefore changes very little on its own. To actually
tailor an application here you have to adapt the on-platform profile - summary,
experience descriptions, skills - which is what :class:`GupyProfileOptimizer`
produces.

The same rule applies as everywhere else: the optimizer may reorder and rephrase
what is already true, and nothing it emits reaches Gupy without passing the fact
validator first.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from cvnator.ai import ChatMessage, ModelRole
from cvnator.ai.gateway import AIGateway
from cvnator.browser.perception import observe
from cvnator.domain.job import ApplyMethod, JobPosting
from cvnator.domain.profile import CandidateMasterProfile
from cvnator.domain.search import SearchProfile
from cvnator.observability import get_logger
from cvnator.platforms.base import PlatformAdapter
from cvnator.resume.ats import ATSAnalysis
from cvnator.resume.fact_validator import FactValidator, ValidationReport

log = get_logger(__name__)


class GupyAdapter(PlatformAdapter):
    name = "gupy"
    domains = ("gupy.io",)
    requires_login = True

    async def discover(self, search: SearchProfile, *, limit: int = 20) -> list[JobPosting]:
        """Gupy job boards are per-company subdomains; there is no global search here.

        Discovery is therefore left to a company career-page crawl or to a posting URL
        supplied by another platform, rather than faked.
        """
        log.info(
            "Gupy has no cross-company search endpoint this adapter can rely on. "
            "Reach Gupy postings via a company board URL or an external-apply handoff."
        )
        return []

    async def open_job(self, job: JobPosting) -> bool:
        return (await self.browser.navigate(job.apply_url or job.url)).ok

    async def enrich(self, job: JobPosting) -> JobPosting:
        state = await observe(self.browser.page)
        job.description = state.text[:20000]
        job.application_platform = self.name
        job.language = job.language or state.lang[:2] or "pt"
        if not job.role:
            job.role = next((e.text.strip() for e in state.elements if e.tag == "h1"), state.title)
        match = re.search(r"/job/(\d+)", state.url)
        if match:
            job.platform_job_id = job.platform_job_id or match.group(1)
        job.apply_method = ApplyMethod.NATIVE_FORM
        return job

    async def is_logged_in(self) -> bool:
        state = await observe(self.browser.page)
        lowered = state.text[:2000].lower()
        return not any(m in lowered for m in ("entrar", "criar conta", "fazer login"))

    async def start_application(self, job: JobPosting) -> ApplyMethod:
        state = await observe(self.browser.page)
        button = next(
            (
                e
                for e in state.elements
                if e.tag in {"button", "a"}
                and re.search(
                    r"candidat|inscrev|quero me candidatar|aplicar",
                    f"{e.text} {e.label}",
                    re.I,
                )
            ),
            None,
        )
        if button is None:
            return ApplyMethod.UNKNOWN
        await self.browser.click(button.selector)
        return ApplyMethod.NATIVE_FORM


# ------------------------------------------------------------- profile optimizer


class GupyProfileEdit(BaseModel):
    """One field of the on-platform profile the agent proposes to change."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="summary | experience | skills | projects")
    target: str = Field(default="", description="Which experience or project, if applicable")
    original: str = Field(description="The current text, copied exactly")
    proposed: str = Field(description="The adapted text; must state the same facts")
    rationale: str = ""


class GupyProfilePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = ""
    summary: str = ""
    skill_order: list[str] = Field(default_factory=list)
    edits: list[GupyProfileEdit] = Field(default_factory=list)
    reasoning: str = ""


_GUPY_SYSTEM = """\
You adapt a candidate's profile fields on a Brazilian job platform (Gupy) to a
specific vacancy. The recruiter reads these fields directly, so they matter as much
as the attached CV.

YOU MAY:
- Rewrite the summary and the experience descriptions to use the vacancy's
  vocabulary and to lead with the most relevant work.
- Reorder skills so the ones the vacancy asks for appear first.

YOU MAY NOT:
- Add a technology, tool, metric, responsibility or result that the original text
  does not already contain.
- Change a company, job title, date, degree or certification.
- Claim a level of experience the profile does not support.

Every `original` you quote must be copied exactly from the profile given to you.
Write in Portuguese unless the vacancy is written in another language.
Return only JSON matching the schema.
"""


class GupyProfileOptimizer:
    """Adapts the on-platform profile, then holds every edit to the fact validator."""

    def __init__(self, gateway: AIGateway) -> None:
        self.gateway = gateway

    async def plan(
        self, profile: CandidateMasterProfile, job: JobPosting, analysis: ATSAnalysis
    ) -> GupyProfilePlan:
        current = [f"SUMMARY: {profile.professional_summary}", "", "EXPERIENCES:"]
        for index, exp in enumerate(profile.experiences):
            current.append(f"[{index}] {exp.job_title} — {exp.company}")
            current += [f"    {b}" for b in (*exp.responsibilities, *exp.achievements)]
        current.append("SKILLS: " + ", ".join(s.name for s in profile.skills))

        messages = [
            ChatMessage(role="system", content=_GUPY_SYSTEM),
            ChatMessage(
                role="user",
                content=(
                    f"CURRENT PROFILE:\n{chr(10).join(current)}\n\n"
                    f"VACANCY: {job.role} at {job.company}\n"
                    f"TECHNOLOGIES SOUGHT: {', '.join(analysis.technologies[:25])}\n"
                    f"KEY TERMS: {', '.join(analysis.keywords[:25])}\n\n"
                    f"DESCRIPTION:\n{job.description[:5000]}"
                ),
            ),
        ]
        return await self.gateway.structured(
            ModelRole.RESUME, GupyProfilePlan, messages, temperature=0.2, max_tokens=4096
        )

    def validate(
        self, profile: CandidateMasterProfile, plan: GupyProfilePlan
    ) -> ValidationReport:
        validator = FactValidator(profile)
        report = ValidationReport(checks_run=["gupy_profile_edits"])
        if plan.summary:
            report.merge(validator.validate_text(plan.summary, context="gupy summary"))
        if plan.headline:
            report.merge(validator.validate_text(plan.headline, context="gupy headline"))

        known = {s.name.strip().lower() for s in profile.skills}
        for skill in plan.skill_order:
            if skill.strip().lower() not in known:
                from cvnator.resume.fact_validator import Severity, Violation, ViolationKind

                report.add(
                    Violation(
                        kind=ViolationKind.UNSUPPORTED_SKILL,
                        severity=Severity.BLOCK,
                        message=f"Plan lists skill '{skill}', which is not in the profile",
                        context="gupy skill_order",
                        evidence=skill,
                    )
                )
        for edit in plan.edits:
            report.merge(
                validator.validate_rewrite(
                    edit.original, edit.proposed, context=f"gupy {edit.field} {edit.target}"
                )
            )
        return report

    async def optimize(
        self, profile: CandidateMasterProfile, job: JobPosting, analysis: ATSAnalysis
    ) -> tuple[GupyProfilePlan, ValidationReport]:
        plan = await self.plan(profile, job, analysis)
        report = self.validate(profile, plan)
        if not report.ok:
            log.warning("Gupy profile plan rejected by the fact validator: %s", report.summary())
        return plan, report


# Not registered on import; see the note in linkedin.py.

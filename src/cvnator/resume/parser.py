"""CV text -> Candidate Master Profile.

The extraction prompt is written defensively: the model is told, repeatedly and
concretely, that leaving a field empty is correct behaviour and inventing one is a
failure. The profile it produces is the system's source of truth, so an invented
fact here would poison every downstream resume and form answer.
"""

from __future__ import annotations

from pathlib import Path

from cvnator.ai import ChatMessage, ModelRole
from cvnator.ai.gateway import AIGateway
from cvnator.domain.profile import CandidateMasterProfile
from cvnator.observability import get_logger
from cvnator.resume.extract import extract_text, sha256_file

log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You extract structured data from a resume/CV. You are a parser, not a writer.

ABSOLUTE RULES:
1. Copy information from the CV. Never invent, infer, embellish or "fill in" anything.
2. If a field is not stated in the CV, leave it empty ("" or [] or null). An empty
   field is a CORRECT answer. A guessed field is a FAILURE.
3. Do not translate company names, job titles, technologies or proper nouns.
4. Do not compute years of experience unless the CV states a number explicitly.
   Leave years_of_experience null and let the dates speak for themselves.
5. Do not add skills that merely seem implied by a job title. Only list skills,
   tools and technologies that literally appear in the CV text.
6. work_authorization and salary_preferences: fill ONLY if explicitly stated.
7. Dates: use "YYYY-MM" when month and year are known, "YYYY" when only the year is
   known, and "" for an ongoing role's end_date (also set is_current true).
8. Split each role's bullet points between `responsibilities` (what they did) and
   `achievements` (a stated result or measurable outcome). Copy the wording.
9. `skills[].category` is "hard" for tools/technologies/methods and "soft" for
   interpersonal traits. `skills[].evidence` lists the company or project names from
   this CV where that skill appears; leave it empty if the CV does not connect them.

Return only JSON matching the schema.
"""


async def parse_cv_text(
    gateway: AIGateway,
    cv_text: str,
    *,
    source_path: str = "",
) -> CandidateMasterProfile:
    """Run the resume model over raw CV text and return a validated profile."""
    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=f"Extract the structured profile from this CV:\n\n---\n{cv_text}\n---",
        ),
    ]
    profile = await gateway.structured(
        ModelRole.RESUME,
        CandidateMasterProfile,
        messages,
        temperature=0.0,
        max_tokens=8192,
    )
    profile.master_cv_text = cv_text
    profile.master_cv_path = source_path
    _backfill_skill_evidence(profile)
    log.info(
        "Parsed CV into profile: %s experiences, %s skills, %s projects",
        len(profile.experiences),
        len(profile.skills),
        len(profile.projects),
    )
    return profile


def _backfill_skill_evidence(profile: CandidateMasterProfile) -> None:
    """Link each skill to the roles/projects that mention it.

    Done deterministically rather than asking the model, because this is exactly the
    kind of cross-referencing a model likes to guess at, and the fact validator
    later relies on the links being true.
    """
    for skill in profile.skills:
        if skill.evidence:
            continue
        needle = skill.name.strip().lower()
        if not needle:
            continue
        evidence: list[str] = []
        for exp in profile.experiences:
            haystack = " ".join(
                [*exp.technologies, *exp.responsibilities, *exp.achievements, exp.job_title]
            ).lower()
            if needle in haystack:
                evidence.append(exp.company)
        for proj in profile.projects:
            haystack = " ".join([*proj.technologies, proj.description, *proj.highlights]).lower()
            if needle in haystack:
                evidence.append(proj.name)
        skill.evidence = evidence


async def parse_cv_file(gateway: AIGateway, path: Path) -> tuple[CandidateMasterProfile, str]:
    """Parse a CV file. Returns the profile and the file's SHA-256."""
    text = extract_text(path)
    digest = sha256_file(path)
    profile = await parse_cv_text(gateway, text, source_path=str(path))
    return profile, digest

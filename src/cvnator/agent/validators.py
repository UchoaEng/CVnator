"""Pre- and post-submission validation.

**Before submitting**, the agent must prove the application is the one it thinks it
is and that it is complete: right job, right company, right resume, required fields
filled, no visible error, no prior submission. This runs on the *observed page*, not
on the agent's beliefs about the page, because those are what go wrong.

**After submitting**, an application is only marked SUBMITTED when there is real
evidence: a confirmation phrase, a reference number, a URL that moved to a
confirmation route, or the form disappearing entirely. Absent evidence the state
becomes SUBMITTING-unconfirmed and the user is told, rather than the system
recording a success it cannot demonstrate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cvnator.browser.perception import PageState
from cvnator.domain.job import JobPosting
from cvnator.observability import get_logger
from cvnator.resume.fact_validator import FactValidator

log = get_logger(__name__)

CONFIRMATION_PATTERNS = (
    r"application (has been )?(successfully )?(submitted|received|sent)",
    r"thank(s| you) for (applying|your application|your interest)",
    r"we('| ha)?ve received your application",
    r"your application (is|was) (complete|submitted|received)",
    r"candidatura (enviada|recebida|realizada|concluída|concluida)",
    r"inscrição (enviada|realizada|recebida|concluída|concluida)",
    r"obrigad[oa] por (se candidatar|sua candidatura|seu interesse)",
    r"recebemos sua candidatura",
    r"(solicitud|candidatura) (enviada|recibida)",
    r"gracias por (postular|tu aplicación|tu candidatura)",
    r"application id",
    r"reference number",
    r"n[úu]mero de refer[êe]ncia",
    r"protocolo",
)
_CONFIRMATION_RE = re.compile("|".join(CONFIRMATION_PATTERNS), re.IGNORECASE)

# Confirmation pages label the reference in several ways: "Reference number: X",
# "Application ID: X", "Protocolo nº X". Two details keep this from matching prose:
# the keyword half is case-insensitive but the captured value is not, and the value
# must contain a digit. Without those, an earlier version happily "found" references
# called "number" and "submitted".
_REFERENCE_RE = re.compile(
    r"(?i:\b(?:ref(?:erence)?|application|candidatura|inscri[çc][ãa]o|protocolo|protocol)"
    r"(?:\s+(?:id|number|num|no|nr|c[óo]digo|n[úu]mero))?"
    r"(?:\s+(?:is|was|é|e))?)"
    r"[\s:#.\-]{0,6}(?:n[ºo°]?[\s.:-]*)?"
    r"(?=[A-Z0-9-]{5,25}\b)(?=[A-Z0-9-]*\d)([A-Z0-9][A-Z0-9-]{4,24})\b"
)

_ALREADY_APPLIED_RE = re.compile(
    r"(you have already applied|already submitted an application|"
    r"você já se candidatou|ja se candidatou|candidatura já (existe|enviada))",
    re.IGNORECASE,
)


@dataclass(slots=True)
class PreSubmissionResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "checks": self.checks,
        }


@dataclass(slots=True)
class SubmissionEvidence:
    confirmed: bool
    signals: list[str] = field(default_factory=list)
    reference: str = ""
    url_after: str = ""
    already_applied: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "confirmed": self.confirmed,
            "signals": self.signals,
            "reference": self.reference,
            "url_after": self.url_after,
            "already_applied": self.already_applied,
        }


def validate_before_submit(
    state: PageState,
    job: JobPosting,
    *,
    resume_path: str = "",
    resume_uploaded: bool = False,
    answers: dict[str, str] | None = None,
    fact_validator: FactValidator | None = None,
) -> PreSubmissionResult:
    """Everything that must hold before the agent is allowed near a submit button."""
    result = PreSubmissionResult(passed=True)
    answers = answers or {}
    page_text = state.text.lower()

    result.checks.append("identity")
    company = (job.company or "").strip().lower()
    role = (job.role or "").strip().lower()
    if company and company not in page_text and company.split()[0] not in page_text:
        result.warnings.append(
            f"Company '{job.company}' does not appear on the page; this may be the wrong "
            "application or a page that simply does not repeat the name"
        )
    if role and not any(word in page_text for word in role.split() if len(word) > 3):
        result.warnings.append(f"Role '{job.role}' does not appear on the page")

    result.checks.append("no_prior_submission")
    if _ALREADY_APPLIED_RE.search(state.text):
        result.passed = False
        result.failures.append("The page states an application already exists for this job")

    result.checks.append("no_visible_errors")
    if state.alerts:
        result.passed = False
        result.failures.append(f"Validation messages present: {'; '.join(state.alerts[:5])}")

    result.checks.append("required_fields")
    for element in state.elements:
        if not element.required or element.disabled:
            continue
        if element.type in {"checkbox", "radio"}:
            if element.checked is False and element.type == "checkbox":
                result.passed = False
                result.failures.append(
                    f"Required checkbox '{element.label or element.name}' is not ticked"
                )
            continue
        if element.type == "file":
            continue
        if element.tag in {"input", "textarea", "select"} and not (element.value or "").strip():
            result.passed = False
            result.failures.append(
                f"Required field '{element.label or element.name or element.element_id}' is empty"
            )

    result.checks.append("resume_uploaded")
    has_file_input = any(e.type == "file" for e in state.elements)
    if has_file_input and not resume_uploaded:
        result.passed = False
        result.failures.append("The form has a file input but no resume was uploaded")
    if resume_uploaded and resume_path:
        from pathlib import Path

        if not Path(resume_path).exists():
            result.passed = False
            result.failures.append(
                f"The resume file {Path(resume_path).name} no longer exists on disk"
            )

    result.checks.append("factual_answers")
    if fact_validator is not None:
        for label, value in answers.items():
            report = fact_validator.validate_text(value, context=label)
            if not report.ok:
                result.passed = False
                result.failures.append(f"Answer to '{label}' is unsupported: {report.summary()}")

    result.checks.append("submit_control_present")
    submit_like = [
        e
        for e in state.elements
        if e.tag == "button" or e.type in {"submit", "button"} or e.role == "button"
    ]
    if not submit_like:
        result.passed = False
        result.failures.append("No submit control found on the page")

    if not result.passed:
        log.warning("Pre-submission validation failed: %s", "; ".join(result.failures))
    return result


def validate_after_submit(
    state_before: PageState, state_after: PageState
) -> SubmissionEvidence:
    """Decide whether there is real evidence that the application went through."""
    evidence = SubmissionEvidence(confirmed=False, url_after=state_after.url)
    text = state_after.text

    if _ALREADY_APPLIED_RE.search(text):
        evidence.already_applied = True
        evidence.signals.append("page reports an existing application")

    match = _CONFIRMATION_RE.search(text)
    if match:
        evidence.signals.append(f"confirmation phrase: '{match.group(0)[:60]}'")

    reference = _REFERENCE_RE.search(text)
    if reference:
        evidence.reference = reference.group(1)
        evidence.signals.append(f"reference number {evidence.reference}")

    url_before = state_before.url.split("?")[0]
    url_after = state_after.url.split("?")[0]
    if url_after != url_before:
        evidence.signals.append(f"URL changed to {url_after}")
        if re.search(
            r"(confirm|success|thank|done|complete|submitted|obrigado|sucesso|conclu)",
            url_after,
            re.IGNORECASE,
        ):
            evidence.signals.append("URL looks like a confirmation route")

    form_fields_before = sum(
        1 for e in state_before.elements if e.tag in {"input", "textarea", "select"}
    )
    form_fields_after = sum(
        1 for e in state_after.elements if e.tag in {"input", "textarea", "select"}
    )
    if form_fields_before >= 3 and form_fields_after == 0:
        evidence.signals.append("the application form is no longer present")

    # Confirmation requires a signal that can ONLY mean "this was submitted": an
    # explicit confirmation phrase, or a reference number.
    #
    # An earlier version also accepted "any two weaker signals", which is unsound and
    # produced a false positive in a real run: advancing from the questions step to
    # the review step changes the URL and removes the inputs, so a page that had
    # submitted nothing was recorded as SUBMITTED. Claiming a submission that did not
    # happen is worse than failing to notice one that did, because the user stops
    # checking. Weak signals are recorded as evidence but never confirm on their own.
    evidence.confirmed = (
        (bool(match) or bool(evidence.reference)) and not evidence.already_applied
    )

    if evidence.confirmed:
        log.info("Submission confirmed by: %s", "; ".join(evidence.signals))
    else:
        log.warning(
            "Submission NOT confirmed. Signals observed: %s",
            "; ".join(evidence.signals) or "none",
        )
    return evidence

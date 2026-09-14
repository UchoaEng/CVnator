"""The Policy Engine.

A layer that sits between the reasoning model and the tools, and that the model
cannot argue with. The model *proposes*; this decides. It is ordinary Python with no
LLM in it, which is the entire point: a prompt can be talked around, a conditional
cannot.

It enforces, in order:

1. **Prohibitions** - things the system will not do at all, such as interacting with
   a CAPTCHA or bypassing an authentication challenge.
2. **Factual integrity** - any value about to be typed into a real application is
   checked against the Candidate Master Profile. An unsupported claim is blocked,
   and the agent is told to ask the user rather than to try a different wording.
3. **Authorization** - high-risk actions (account creation, legal consent, final
   submission) run only within the autonomy settings the user configured.
4. **Pacing** - per-platform rate limits and daily caps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cvnator.domain.actions import (
    ActionType,
    AgentDecision,
    PolicyOutcome,
    PolicyVerdict,
    RiskLevel,
)
from cvnator.domain.application import InterventionKind
from cvnator.domain.profile import CandidateMasterProfile
from cvnator.domain.search import AutonomySettings
from cvnator.observability import get_logger
from cvnator.resume.fact_validator import FactValidator

log = get_logger(__name__)

#: Markers that a page is presenting a human-verification or MFA challenge.
_CHALLENGE_PATTERNS = (
    r"\brecaptcha\b", r"\bhcaptcha\b", r"\bcaptcha\b", r"\bcloudflare\b",
    r"\bturnstile\b", r"are you (a )?human", r"verify you are human",
    r"não sou um robô", r"nao sou um robo", r"i'?m not a robot",
    r"two[- ]factor", r"\bmfa\b", r"\b2fa\b", r"authentication code",
    r"verification code", r"c[óo]digo de verifica[çc][ãa]o", r"one[- ]time password",
    r"\botp\b", r"passkey", r"security key",
)
_CHALLENGE_RE = re.compile("|".join(_CHALLENGE_PATTERNS), re.IGNORECASE)


def detect_challenge(page_text: str) -> str:
    """Return the matched challenge marker, or an empty string."""
    match = _CHALLENGE_RE.search(page_text or "")
    return match.group(0) if match else ""


@dataclass(slots=True)
class PolicyContext:
    """Everything the policy engine needs to rule on a decision."""

    autonomy: AutonomySettings
    profile: CandidateMasterProfile
    page_text: str = ""
    platform: str = ""
    applications_today: int = 0
    applications_this_session: int = 0
    has_stored_credentials: bool = False
    pre_submission_passed: bool = False
    field_label: str = ""
    known_answers: dict[str, str] = field(default_factory=dict)


class PolicyEngine:
    def __init__(self, profile: CandidateMasterProfile) -> None:
        self.profile = profile
        self.validator = FactValidator(profile)

    # ------------------------------------------------------------------- evaluation

    def evaluate(self, decision: AgentDecision, context: PolicyContext) -> PolicyVerdict:
        for check in (
            self._check_prohibited_challenges,
            self._check_dry_run,
            self._check_limits,
            self._check_factual_integrity,
            self._check_high_risk_authorization,
        ):
            verdict = check(decision, context)
            if verdict is not None:
                if verdict.outcome is not PolicyOutcome.ALLOW:
                    log.decision(
                        "POLICY %s on %s [%s]: %s",
                        verdict.outcome.value.upper(),
                        decision.action.value,
                        verdict.rule,
                        verdict.reason,
                    )
                return verdict
        return PolicyVerdict(outcome=PolicyOutcome.ALLOW, rule="default_allow")

    # ------------------------------------------------------------------------ rules

    def _check_prohibited_challenges(
        self, decision: AgentDecision, context: PolicyContext
    ) -> PolicyVerdict | None:
        """Never attempt to solve or work around a human-verification challenge."""
        if decision.risk is RiskLevel.LOW and decision.action not in {
            ActionType.CLICK_ELEMENT,
            ActionType.FILL_FIELD,
        }:
            return None

        marker = detect_challenge(context.page_text)
        if not marker:
            return None

        kind = (
            InterventionKind.CAPTCHA
            if re.search(r"captcha|human|robot|robo|turnstile|cloudflare", marker, re.I)
            else InterventionKind.MFA
        )
        return PolicyVerdict(
            outcome=PolicyOutcome.ASK_USER,
            rule="no_challenge_bypass",
            reason=(
                f"The page is presenting a verification challenge ('{marker}'). "
                "CVnator does not solve or circumvent these. The user must complete it."
            ),
            intervention_kind=kind.value,
        )

    def _check_dry_run(
        self, decision: AgentDecision, context: PolicyContext
    ) -> PolicyVerdict | None:
        if not context.autonomy.dry_run:
            return None
        if decision.action in {
            ActionType.SUBMIT_APPLICATION,
            ActionType.CREATE_ACCOUNT,
            ActionType.ACCEPT_LEGAL_TERMS,
        }:
            return PolicyVerdict(
                outcome=PolicyOutcome.BLOCK,
                rule="dry_run",
                reason=(
                    f"DRY RUN is enabled, so {decision.action.value} is not performed. "
                    "Everything up to this point was carried out for real."
                ),
            )
        return None

    def _check_limits(
        self, decision: AgentDecision, context: PolicyContext
    ) -> PolicyVerdict | None:
        if decision.action is not ActionType.SUBMIT_APPLICATION:
            return None
        autonomy = context.autonomy
        if context.applications_today >= autonomy.daily_application_limit:
            return PolicyVerdict(
                outcome=PolicyOutcome.BLOCK,
                rule="daily_limit",
                reason=(
                    f"Daily application limit reached "
                    f"({context.applications_today}/{autonomy.daily_application_limit})"
                ),
            )
        if context.applications_this_session >= autonomy.session_application_limit:
            return PolicyVerdict(
                outcome=PolicyOutcome.BLOCK,
                rule="session_limit",
                reason=(
                    f"Session application limit reached "
                    f"({context.applications_this_session}/"
                    f"{autonomy.session_application_limit})"
                ),
            )
        return None

    def _check_factual_integrity(
        self, decision: AgentDecision, context: PolicyContext
    ) -> PolicyVerdict | None:
        """Anything about to be typed into a real application must be defensible."""
        if decision.action not in {
            ActionType.FILL_FIELD,
            ActionType.TYPE_TEXT,
            ActionType.ANSWER_QUESTION,
            ActionType.SELECT_OPTION,
        }:
            return None
        value = (decision.value or "").strip()
        if not value:
            return None

        label = context.field_label or decision.target.text or decision.target.element_id
        report = self.validator.validate_text(value, context=label)

        # A field explicitly asking for a duration deserves the stricter numeric check
        # even when the answer is a bare number ("3" in a "years of SQL" box).
        numeric_claim = self._numeric_field_claim(label, value)
        if numeric_claim is not None:
            report.violations.extend(numeric_claim)

        if report.ok:
            return None

        return PolicyVerdict(
            outcome=PolicyOutcome.BLOCK,
            rule="fact_integrity",
            reason=(
                f"Value rejected as unsupported by the candidate profile: "
                f"{report.summary()}. Do not rephrase it - either use a supported "
                f"value or request the answer from the user."
            ),
            intervention_kind=InterventionKind.UNKNOWN_PERSONAL_INFORMATION.value,
        )

    def _numeric_field_claim(self, label: str, value: str):
        """Check a bare number typed into a "years of X" field.

        The number carries no claim on its own; the label is what turns "6" into an
        assertion. Identifying *which* skill the question is about matters: a greedy
        subject match on "How many years of experience do you have with SQL?" yields
        "experience do you have with sql", which matches no skill and silently falls
        back to total experience - letting a candidate with 5 years of SQL answer 6.
        """
        from cvnator.resume.fact_validator import Severity, Violation, ViolationKind
        from cvnator.resume.vocabulary import find_tech_terms

        if not re.fullmatch(r"\d+([.,]\d+)?", value.strip()):
            return None
        years_field = re.search(
            r"(?:years?|anos?|años?)\s+(?:of|with|de|em|com)?\s*([\w .+#/-]{2,40})",
            label,
            re.IGNORECASE,
        )
        if not years_field:
            return None
        subject = years_field.group(1).strip().rstrip("?*:").lower()
        claimed = float(value.replace(",", "."))

        # Prefer a named technology in the label over the free-text subject.
        supported: float | None = None
        for term in sorted(find_tech_terms(label), key=len, reverse=True):
            supported = self.profile.years_with_skill(term)
            if supported is not None:
                subject = term
                break
        if supported is None:
            supported = self.profile.years_with_skill(subject)
        if supported is None:
            supported = self.profile.total_years_experience()
        if claimed > supported + 0.6:
            return [
                Violation(
                    kind=ViolationKind.INFLATED_DURATION,
                    severity=Severity.BLOCK,
                    message=(
                        f"Field asks for years of '{subject}'; answering {claimed:g} "
                        f"exceeds the {supported:g} the profile supports. "
                        f"Enter {supported:g} instead."
                    ),
                    context=label,
                    evidence=value,
                )
            ]
        return None

    def _check_high_risk_authorization(
        self, decision: AgentDecision, context: PolicyContext
    ) -> PolicyVerdict | None:
        autonomy = context.autonomy

        if decision.action is ActionType.CREATE_ACCOUNT:
            if autonomy.account_policy == "never":
                return PolicyVerdict(
                    outcome=PolicyOutcome.BLOCK,
                    rule="account_policy_never",
                    reason="Account creation is disabled. Abandon this application.",
                )
            if autonomy.account_policy == "auto" and autonomy.auto_create_external_account:
                return PolicyVerdict(outcome=PolicyOutcome.ALLOW, rule="account_policy_auto")
            return PolicyVerdict(
                outcome=PolicyOutcome.ASK_USER,
                rule="account_policy_ask",
                reason=(
                    f"Creating an account on {context.platform or 'this platform'} requires "
                    "the user's authorization."
                ),
                intervention_kind=InterventionKind.ACCOUNT_CREATION_APPROVAL.value,
            )

        if decision.action is ActionType.LOGIN:
            if not context.has_stored_credentials:
                return PolicyVerdict(
                    outcome=PolicyOutcome.ASK_USER,
                    rule="login_no_credentials",
                    reason=(
                        f"No stored credentials for {context.platform or 'this platform'}. "
                        "The user must sign in, or authorize storing credentials."
                    ),
                    intervention_kind=InterventionKind.LOGIN_REQUIRED.value,
                )
            return PolicyVerdict(outcome=PolicyOutcome.ALLOW, rule="login_with_vault")

        if decision.action is ActionType.ACCEPT_LEGAL_TERMS:
            return PolicyVerdict(
                outcome=PolicyOutcome.ASK_USER,
                rule="legal_consent",
                reason=(
                    "Accepting terms or giving legal consent is the user's decision, "
                    "not the agent's."
                ),
                intervention_kind=InterventionKind.LEGAL_CONSENT.value,
            )

        if decision.action is ActionType.SUBMIT_APPLICATION:
            if not context.pre_submission_passed:
                return PolicyVerdict(
                    outcome=PolicyOutcome.BLOCK,
                    rule="submission_requires_validation",
                    reason=(
                        "Pre-submission validation has not passed. Run "
                        "validate_before_submit first."
                    ),
                )
            if autonomy.review_before_submit or not autonomy.auto_submit:
                return PolicyVerdict(
                    outcome=PolicyOutcome.ASK_USER,
                    rule="review_before_submit",
                    reason="The user has asked to review applications before they are sent.",
                    intervention_kind=InterventionKind.SUBMIT_APPROVAL.value,
                )
            return PolicyVerdict(outcome=PolicyOutcome.ALLOW, rule="auto_submit_enabled")

        return None

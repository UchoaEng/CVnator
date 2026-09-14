"""The agent loop.

    OBSERVE -> UNDERSTAND -> PLAN -> ACT -> VALIDATE -> UPDATE MEMORY -> CONTINUE

The model decides; tools execute; the loop validates and recovers. Concretely, each
cycle:

* **OBSERVE** - read the page through the perception layers, cheapest first.
* **UNDERSTAND** - assemble a bounded context: the goal, the current checkpoint, the
  page inventory, remembered answers that apply here, and the *result of the last
  action*. Not the whole history - see :meth:`FormAgent._build_context`.
* **PLAN** - the reasoning model emits one schema-validated ``AgentDecision``.
* **ACT** - the policy engine rules on it, then a tool runs it.
* **VALIDATE** - the tool's own report says whether the page really moved. The
  watchdog checks for loops. Failures walk the recovery ladder.
* **UPDATE MEMORY** - checkpoints, answers learned from the user, platform quirks.

Escalation to the user is a first-class outcome, not an error path.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvnator.agent.memory import MemoryScope, MemoryStore
from cvnator.agent.policy import PolicyContext, PolicyEngine, detect_challenge
from cvnator.agent.validators import (
    PreSubmissionResult,
    SubmissionEvidence,
    validate_after_submit,
    validate_before_submit,
)
from cvnator.agent.watchdog import RecoveryStrategy, Watchdog
from cvnator.ai import ChatMessage, ModelRole
from cvnator.ai.gateway import AIGateway
from cvnator.browser.perception import (
    PageState,
    accessibility_snapshot,
    flatten_accessibility,
    needs_visual_fallback,
)
from cvnator.browser.tools import BrowserToolbox
from cvnator.browser.vision import VisualReading, read_screenshot
from cvnator.domain.actions import (
    ActionType,
    AgentDecision,
    ElementTarget,
    PolicyOutcome,
    ToolResult,
)
from cvnator.domain.application import (
    ApplicationState,
    Checkpoint,
    InterventionKind,
    UserActionRequest,
)
from cvnator.domain.job import JobPosting
from cvnator.domain.profile import CandidateMasterProfile
from cvnator.domain.search import AutonomySettings
from cvnator.observability import get_logger
from cvnator.resume.fact_validator import FactValidator

log = get_logger(__name__)

#: Signature: (request) -> answer text, or None if the user declined / is unavailable.
AskUser = Callable[[UserActionRequest], Awaitable[str | None]]

#: Actions for which "the page did not change" is genuine evidence of failure.
#: Filling a text input is not one of them.
_PAGE_CHANGING_ACTIONS = frozenset(
    {
        ActionType.CLICK_ELEMENT,
        ActionType.NAVIGATE,
        ActionType.GO_BACK,
        ActionType.PRESS_KEY,
        ActionType.SUBMIT_APPLICATION,
        ActionType.SWITCH_TAB,
        ActionType.OPEN_TAB,
    }
)

_AGENT_SYSTEM = """\
You are the decision-making core of an autonomous job application agent. You control a
real browser through tools. You emit exactly ONE action per turn as JSON.

HOW TO THINK
- You are given the current page: its URL, its interactive elements (each with a
  stable id like e12), any validation messages, and its visible text.
- Target elements by their `element_id`. Only fall back to `target.text` if the id
  you want is not listed.
- Read the RESULT OF YOUR LAST ACTION carefully, and trust `success` over your own
  expectation. A successful fill does NOT change the page - that is normal. Never
  refill a field the page already shows as filled; move on to the next empty one.
- A click that reports success:false with "the page did not change" really did fail.
  Do something different rather than clicking it again.
- Validation messages on the page are the page telling you what is wrong. Fix the
  named field rather than clicking Continue again.
- Element lines show `current="..."` when a field already holds a value, and
  `REQUIRED` when it must be filled. Work through the empty required fields first.
- Use `select_option` for a dropdown (an element showing `options=[...]`) and
  `set_checkbox` for a checkbox or radio. `fill_field` is for text inputs only.
- A file input (`input:file`) can NEVER be filled with text. Use `upload_file`
  targeting it. If the context says a tailored resume is ready, that file is used
  automatically - do not ask the user for a path.
- A field may show `constraints(...)` and `INVALID: "..."`. That is the browser
  refusing the value, and it will silently block the whole form from submitting.
- When a constraint makes the exactly truthful value impossible to enter (a `step`
  that rejects 6.2, a `maxlength` that truncates), enter the nearest permitted value
  that does NOT overstate the candidate - round DOWN, never up. 6.2 years with a
  step of 0.5 becomes 6, not 6.5. Then move on; never oscillate between two values.

HARD RULES
- Never state a fact about the candidate that is not in the profile facts you are
  given. If a form asks something you cannot answer truthfully from the profile, use
  `request_user_input`. Guessing is the worst possible action.
- Never attempt to solve, bypass or click through a CAPTCHA, MFA prompt or any human
  verification. Use `request_user_input` with a clear explanation.
- Never type a password. Use the `login` action; credentials are handled outside your
  context and you will never see them.
- `submit_application` is only valid after `validate_before_submit` has passed.
- If the application is finished, use `finish`. If it cannot be completed, use
  `abandon` with a reason. Do not loop.

ANSWERING QUESTIONS
- Prefer a remembered answer when one is listed for this question.
- Answer in the language of the form.
- Keep free-text answers concrete and drawn from the candidate's real experience.
  Two to four sentences.

OPEN QUESTIONS LIKE "Why do you want to work here?"
These are answerable and you should answer them. The rule against inventing facts
applies to claims about THE CANDIDATE, not to ordinary motivation.
- Build the answer from two things you already have: what the posting says the role
  involves, and what the candidate has actually done.
- Do not assert facts about the company beyond what the posting states. Do not
  praise its culture, size, mission or products unless the posting says so.
- Do not claim prior familiarity with the company, or that they have "long admired"
  it. Connect the role's stated responsibilities to the candidate's real work.
- Only use `request_user_input` here if the question asks for something factual you
  genuinely do not have - a salary figure, a notice period, a certification. Not
  knowing anything about the employer is NOT a reason to escalate.

Return only JSON matching the schema.
"""


@dataclass(slots=True)
class AgentCycle:
    """One turn of the loop, recorded for audit and for the UI's activity view."""

    index: int
    current_goal: str
    current_state: str
    page_url: str
    page_fingerprint: str
    available_actions: int
    last_action: str = ""
    expected_result: str = ""
    observed_result: str = ""
    retry_count: int = 0
    decision_summary: str = ""
    policy_outcome: str = ""
    success: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "goal": self.current_goal,
            "state": self.current_state,
            "url": self.page_url,
            "decision": self.decision_summary,
            "policy": self.policy_outcome,
            "expected": self.expected_result,
            "observed": self.observed_result[:400],
            "retry_count": self.retry_count,
            "success": self.success,
        }


@dataclass
class ApplicationOutcome:
    state: ApplicationState
    checkpoints: list[Checkpoint] = field(default_factory=list)
    cycles: list[AgentCycle] = field(default_factory=list)
    answers_used: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    evidence: SubmissionEvidence | None = None
    pre_submission: PreSubmissionResult | None = None
    pending_user_action: UserActionRequest | None = None
    vision_fallbacks: int = 0
    tool_retries: int = 0

    @property
    def submitted(self) -> bool:
        return self.state is ApplicationState.SUBMITTED


class FormAgent:
    """Drives a single application from "form is open" to a terminal state."""

    def __init__(
        self,
        gateway: AIGateway,
        toolbox: BrowserToolbox,
        policy: PolicyEngine,
        memory: MemoryStore,
        profile: CandidateMasterProfile,
        autonomy: AutonomySettings,
        *,
        ask_user: AskUser | None = None,
        max_steps: int = 60,
        loop_threshold: int = 4,
    ) -> None:
        self.gateway = gateway
        self.toolbox = toolbox
        self.policy = policy
        self.memory = memory
        self.profile = profile
        self.autonomy = autonomy
        self.ask_user = ask_user
        self.fact_validator = FactValidator(profile)
        self.watchdog = Watchdog(loop_threshold=loop_threshold, max_steps=max_steps)

    # ------------------------------------------------------------------ context

    def _profile_digest(self) -> str:
        """A compact, factual briefing. Bounded so context never runs away."""
        info = self.profile.personal_information
        lines = [
            f"Name: {info.full_name}",
            f"Email: {info.email}",
            f"Phone: {info.phone}",
            f"Location: {', '.join(p for p in (info.city, info.state, info.country) if p)}",
            f"LinkedIn: {info.linkedin or '(not provided)'}",
            f"GitHub: {info.github or '(not provided)'}",
            f"Total years of experience on file: {self.profile.total_years_experience()}",
        ]
        if self.profile.experiences:
            current = self.profile.experiences[0]
            lines.append(f"Most recent role: {current.job_title} at {current.company}")
        # Years per skill matter: without them the agent cannot answer "how many years
        # of SQL?" truthfully and escalates to the user for data already on file.
        skill_bits: list[str] = []
        for skill in self.profile.skills[:40]:
            years = skill.years_of_experience
            if years is None:
                years = self.profile.years_with_skill(skill.name)
            skill_bits.append(f"{skill.name} ({years:g}y)" if years is not None else skill.name)
        lines.append("Skills: " + ", ".join(skill_bits))
        lines.append(
            "A skill shown with (Ny) means N years of experience with it - use that "
            "number when a form asks, never the total."
        )
        if self.profile.languages:
            lines.append(
                "Languages: "
                + ", ".join(
                    f"{lang.language}={lang.proficiency.value}" for lang in self.profile.languages
                )
            )
        auth = self.profile.work_authorization
        if auth.countries_authorized or auth.visa_status:
            lines.append(
                "Work authorization: "
                + ", ".join(auth.countries_authorized)
                + (f" ({auth.visa_status})" if auth.visa_status else "")
            )
        else:
            lines.append(
                "Work authorization: NOT ON FILE - you must ask the user, never assume"
            )
        salary = self.profile.salary_preferences
        if salary.target or salary.minimum:
            lines.append(
                f"Salary expectation: {salary.currency} "
                f"{salary.target or salary.minimum:g}/{salary.period}"
            )
        return "\n".join(lines)

    def _build_context(
        self,
        state: PageState,
        job: JobPosting,
        goal: str,
        checkpoint: Checkpoint | None,
        last_result: ToolResult | None,
        recovery_hint: str,
        visual: VisualReading | None,
        scope: MemoryScope,
        resume_path: Path | None = None,
        settled: dict[str, str] | None = None,
        corrections: dict[str, str] | None = None,
    ) -> str:
        """Bounded per-step context.

        Deliberately excludes the full action history: the loop's own validation
        already tells the model whether the last action worked, and replaying dozens
        of prior turns would blow the context window on a 9B local model without
        improving the next decision.
        """
        known = self.memory.known_answers_for(scope, limit=25)
        blocks = [
            f"GOAL: {goal}",
            f"TARGET JOB: {job.role} at {job.company} ({job.location})",
            f"PROGRESS: last checkpoint reached = {checkpoint.value if checkpoint else 'none'}",
            "",
            "CANDIDATE PROFILE FACTS (the only facts you may state):",
            self._profile_digest(),
        ]
        if resume_path is not None:
            # Mentioning the file only when the page can accept one stops the agent
            # trying to upload it three steps early.
            has_file_input = any(e.type == "file" for e in state.elements)
            if has_file_input:
                blocks += [
                    "",
                    f"THIS PAGE HAS A FILE INPUT and a tailored resume is ready "
                    f"({resume_path.name}). Use `upload_file` on it and leave `value` "
                    f"empty - the runtime supplies the file. You never choose files "
                    f"from disk.",
                ]
            else:
                blocks += [
                    "",
                    f"A tailored resume ({resume_path.name}) is ready for the document "
                    f"step. This page has no file input, so there is nothing to upload "
                    f"yet - complete this step first.",
                ]
        if known:
            blocks += [
                "",
                "REMEMBERED ANSWERS (reuse these when the question matches):",
                *[f"  - {q}: {a}" for q, a in list(known.items())[:25]],
            ]
        blocks += ["", "CURRENT PAGE:", state.as_prompt_block()]
        if visual is not None:
            blocks += ["", visual.as_prompt_block()]
        if last_result is not None:
            lines = [
                "",
                "RESULT OF YOUR LAST ACTION:",
                f"  action: {last_result.action.value}",
                f"  success: {last_result.success}",
                f"  error: {last_result.error or '(none)'}",
            ]
            if last_result.action in _PAGE_CHANGING_ACTIONS:
                lines += [
                    f"  url changed: {last_result.changed_url} "
                    f"({last_result.url_before} -> {last_result.url_after})",
                    f"  page changed: {last_result.dom_changed}",
                ]
            else:
                # For a fill or a select, what matters is the value that landed, not
                # whether the page re-rendered.
                value_after = last_result.data.get("value_after")
                if value_after is not None:
                    lines.append(f"  field now contains: {value_after!r}")
                lines.append(
                    "  (this kind of action is not expected to change the page)"
                )
            lines.append(f"  observed: {last_result.observed_state[:600]}")
            blocks += lines
        if settled:
            blocks += [
                "",
                "VALUES YOU HAVE ALREADY ENTERED IN THIS APPLICATION "
                "(settled; do not revisit them):",
                *[f"  - {label}: {value}" for label, value in list(settled.items())[:20]],
            ]
        if corrections:
            blocks += [
                "",
                "CORRECTIONS THE POLICY ENGINE HAS ALREADY REQUIRED (final - "
                "re-deriving the original value will simply be blocked again):",
                *[f"  - {label}: use {value}" for label, value in list(corrections.items())[:10]],
            ]
        if recovery_hint:
            blocks += ["", f"RECOVERY INSTRUCTION: {recovery_hint}"]
        return "\n".join(blocks)

    # --------------------------------------------------------------------- plan

    async def _plan(self, context: str) -> AgentDecision:
        messages = [
            ChatMessage(role="system", content=_AGENT_SYSTEM),
            ChatMessage(role="user", content=context),
        ]
        return await self.gateway.structured(
            ModelRole.REASONING, AgentDecision, messages, temperature=0.1, max_tokens=1536
        )

    # ---------------------------------------------------------------- user input

    async def _escalate(
        self,
        kind: InterventionKind,
        job: JobPosting,
        problem: str,
        action_needed: str,
        question: str = "",
    ) -> str | None:
        request = UserActionRequest(
            kind=kind,
            platform=job.platform,
            job_title=job.role,
            company=job.company,
            problem=problem,
            action_needed=action_needed,
            question=question,
        )
        log.needs_user("[%s] %s | %s", kind.value, problem, action_needed)
        if self.ask_user is None:
            return None
        return await self.ask_user(request)

    # ------------------------------------------------------------------ run loop

    async def run(
        self,
        job: JobPosting,
        *,
        goal: str = "Complete and submit this job application",
        resume_path: Path | None = None,
        applications_today: int = 0,
        applications_this_session: int = 0,
        has_stored_credentials: bool = False,
    ) -> ApplicationOutcome:
        outcome = ApplicationOutcome(state=ApplicationState.APPLYING)
        scope = MemoryScope(
            platform=job.platform,
            role=job.role,
            company=job.company,
            job_id=job.platform_job_id or job.url,
        )
        self.watchdog.reset()

        last_result: ToolResult | None = None
        recovery_hint = ""
        #: field label -> value the policy engine insisted on. Kept for the whole
        #: application so a blocked value is not re-derived a few turns later.
        corrections: dict[str, str] = {}
        visual: VisualReading | None = None
        checkpoint: Checkpoint | None = None
        pre_submission_passed = False
        resume_uploaded = False
        retry_count = 0

        while True:
            state = await self.toolbox.context.refresh()

            # --- OBSERVE: escalate immediately on a verification challenge -------
            challenge = detect_challenge(state.text)
            if challenge:
                kind = (
                    InterventionKind.CAPTCHA
                    if "captcha" in challenge.lower() or "robot" in challenge.lower()
                    else InterventionKind.MFA
                )
                answer = await self._escalate(
                    kind,
                    job,
                    f"The page is showing a verification challenge ('{challenge}').",
                    "Please complete the verification in the open browser, then resume.",
                )
                if answer is None:
                    outcome.state = ApplicationState.WAITING_USER
                    outcome.pending_user_action = UserActionRequest(
                        kind=kind, platform=job.platform, job_title=job.role,
                        company=job.company,
                        problem=f"Verification challenge: {challenge}",
                        action_needed="Complete the challenge in the browser, then resume.",
                    )
                    return outcome
                # The user says they handled it; re-observe rather than assume.
                last_result = None
                recovery_hint = "The user completed a verification challenge. Re-read the page."
                continue

            # --- OBSERVE: vision fallback only when cheaper layers failed --------
            visual = None
            need_vision, why = needs_visual_fallback(state)
            if need_vision or recovery_hint.startswith("VISUAL"):
                try:
                    shot = await self.toolbox.context.browser.screenshot()
                    visual = await read_screenshot(self.gateway, shot)
                    outcome.vision_fallbacks += 1
                    log.info("Vision fallback used: %s", why or "recovery ladder")
                except Exception as exc:
                    log.warning("Vision fallback failed: %s", exc)

            # --- UNDERSTAND + PLAN ----------------------------------------------
            context = self._build_context(
                state, job, goal, checkpoint, last_result, recovery_hint, visual, scope,
                resume_path, outcome.answers_used, corrections,
            )
            try:
                decision = await self._plan(context)
            except Exception as exc:
                outcome.errors.append(f"Planning failed: {exc}")
                outcome.state = ApplicationState.FAILED
                return outcome

            cycle = AgentCycle(
                index=self.watchdog.steps,
                current_goal=goal,
                current_state=outcome.state.value,
                page_url=state.url,
                page_fingerprint=state.fingerprint,
                available_actions=len(state.elements),
                last_action=last_result.action.value if last_result else "",
                expected_result=decision.expected_result,
                retry_count=retry_count,
                decision_summary=decision.summary(),
            )
            log.decision("%s | %s", decision.action.value, decision.reason[:160])

            # --- terminal actions the agent may choose itself ---------------------
            if decision.action is ActionType.FINISH:
                cycle.policy_outcome = "allow"
                cycle.success = True
                outcome.cycles.append(cycle)
                outcome.state = (
                    ApplicationState.SUBMITTED
                    if outcome.evidence and outcome.evidence.confirmed
                    else ApplicationState.APPLYING
                )
                if not (outcome.evidence and outcome.evidence.confirmed):
                    outcome.errors.append(
                        "Agent called finish without confirmed submission evidence"
                    )
                    outcome.state = ApplicationState.FAILED
                return outcome

            if decision.action is ActionType.ABANDON:
                cycle.policy_outcome = "allow"
                outcome.cycles.append(cycle)
                outcome.errors.append(decision.reason or "Agent abandoned the application")
                outcome.state = ApplicationState.ABANDONED
                return outcome

            # --- ACT: policy engine rules first -----------------------------------
            field_label = ""
            if decision.target.element_id:
                element = state.element_by_id(decision.target.element_id)
                field_label = element.label if element else ""

            policy_context = PolicyContext(
                autonomy=self.autonomy,
                profile=self.profile,
                page_text=state.text,
                platform=job.platform,
                applications_today=applications_today,
                applications_this_session=applications_this_session,
                has_stored_credentials=has_stored_credentials,
                pre_submission_passed=pre_submission_passed,
                field_label=field_label,
            )
            verdict = self.policy.evaluate(decision, policy_context)
            cycle.policy_outcome = verdict.outcome.value

            if verdict.outcome is PolicyOutcome.BLOCK:
                outcome.cycles.append(cycle)
                last_result = ToolResult(
                    success=False,
                    action=decision.action,
                    error=f"Blocked by policy [{verdict.rule}]: {verdict.reason}",
                    url_before=state.url,
                    url_after=state.url,
                    page_fingerprint=state.fingerprint,
                )
                suggested = re.search(r"Enter ([\w.,]+) instead", verdict.reason)
                if suggested and field_label:
                    corrections[field_label] = suggested.group(1).rstrip(".")
                recovery_hint = (
                    f"The policy engine blocked that action: {verdict.reason} "
                    "Choose a different action. Do not retry the blocked one."
                )
                retry_count += 1
                if retry_count > 4:
                    outcome.errors.append(f"Policy kept blocking: {verdict.reason}")
                    outcome.state = ApplicationState.FAILED
                    return outcome
                continue

            if verdict.outcome is PolicyOutcome.ASK_USER:
                kind = InterventionKind(verdict.intervention_kind) if verdict.intervention_kind \
                    else InterventionKind.UNKNOWN_PERSONAL_INFORMATION
                answer = await self._escalate(
                    kind,
                    job,
                    verdict.reason,
                    f"Approve or complete: {decision.action.value}",
                    question=decision.value or "",
                )
                outcome.cycles.append(cycle)
                if answer is None:
                    outcome.state = ApplicationState.WAITING_USER
                    outcome.pending_user_action = UserActionRequest(
                        kind=kind, platform=job.platform, job_title=job.role,
                        company=job.company, problem=verdict.reason,
                        action_needed=f"Authorize or perform: {decision.action.value}",
                    )
                    return outcome
                if answer.strip().lower() in {"no", "deny", "cancel", "não", "nao"}:
                    outcome.errors.append(f"User declined: {decision.action.value}")
                    outcome.state = ApplicationState.ABANDONED
                    return outcome
                recovery_hint = f"The user authorized this step and said: {answer}"
                last_result = None
                continue

            # --- ACT: handle actions the browser toolbox does not own -------------
            handled, last_result, checkpoint, pre_submission_passed, resume_uploaded = (
                await self._handle_special_action(
                    decision, state, job, outcome, checkpoint, scope,
                    resume_path, pre_submission_passed, resume_uploaded,
                )
            )
            if handled and last_result is None:
                # The action terminated the run (escalation or hard stop).
                outcome.cycles.append(cycle)
                return outcome

            if not handled:
                last_result = await self.toolbox.execute(decision)
                if decision.action is ActionType.UPLOAD_FILE and last_result.success:
                    resume_uploaded = True
                    checkpoint = Checkpoint.RESUME_UPLOADED
                    if checkpoint not in outcome.checkpoints:
                        outcome.checkpoints.append(checkpoint)

            assert last_result is not None
            cycle.observed_result = last_result.observed_state or last_result.error
            cycle.success = last_result.success
            outcome.cycles.append(cycle)

            # --- VALIDATE + recover ------------------------------------------------
            watchdog_verdict = self.watchdog.record(decision, last_result)
            if not last_result.success:
                outcome.tool_retries += 1
                retry_count += 1
            else:
                retry_count = 0

            recovery_hint = ""
            if watchdog_verdict.reason:
                log.warning("Watchdog: %s -> %s", watchdog_verdict.reason,
                            watchdog_verdict.strategy.value)
                recovery = await self._apply_recovery(
                    watchdog_verdict.strategy, watchdog_verdict.reason, job, outcome
                )
                if recovery is None:
                    outcome.state = (
                        ApplicationState.WAITING_USER
                        if outcome.pending_user_action
                        else ApplicationState.FAILED
                    )
                    return outcome
                recovery_hint = recovery
            elif not last_result.success:
                recovery_hint = (
                    f"That action failed: {last_result.error} "
                    "Re-read the page and try a different approach."
                )

            # --- UPDATE MEMORY -----------------------------------------------------
            if last_result.success and decision.action in {
                ActionType.FILL_FIELD,
                ActionType.SELECT_OPTION,
            } and field_label and decision.value:
                outcome.answers_used[field_label] = decision.value

            await asyncio.sleep(0.2)

    # ------------------------------------------------------------- special actions

    async def _handle_special_action(
        self,
        decision: AgentDecision,
        state: PageState,
        job: JobPosting,
        outcome: ApplicationOutcome,
        checkpoint: Checkpoint | None,
        scope: MemoryScope,
        resume_path: Path | None,
        pre_submission_passed: bool,
        resume_uploaded: bool,
    ) -> tuple[bool, ToolResult | None, Checkpoint | None, bool, bool]:
        """Actions that are not browser primitives. Returns (handled, result, ...)."""
        action = decision.action

        if action is ActionType.REQUEST_USER_INPUT:
            question = decision.value or decision.args.get("question", "")
            remembered = self.memory.recall_answer(str(question), scope)
            if remembered is not None:
                result = ToolResult(
                    success=True,
                    action=action,
                    observed_state=f"Answer recalled from memory: {remembered.value}",
                    url_before=state.url,
                    url_after=state.url,
                    page_fingerprint=state.fingerprint,
                    data={"answer": remembered.value, "source": "memory"},
                )
                return True, result, checkpoint, pre_submission_passed, resume_uploaded

            answer = await self._escalate(
                InterventionKind.UNKNOWN_PERSONAL_INFORMATION,
                job,
                decision.reason or "The agent needs information it cannot derive from the profile.",
                "Answer the question so the application can continue.",
                question=str(question),
            )
            if answer is None:
                outcome.state = ApplicationState.WAITING_USER
                outcome.pending_user_action = UserActionRequest(
                    kind=InterventionKind.UNKNOWN_PERSONAL_INFORMATION,
                    platform=job.platform, job_title=job.role, company=job.company,
                    problem=decision.reason or "Missing information",
                    action_needed="Provide the answer below",
                    question=str(question),
                )
                return True, None, checkpoint, pre_submission_passed, resume_uploaded

            self.memory.remember_answer(str(question), answer, scope="global", source="user")
            outcome.answers_used[str(question)] = answer
            result = ToolResult(
                success=True,
                action=action,
                observed_state=f"The user answered: {answer}",
                url_before=state.url,
                url_after=state.url,
                page_fingerprint=state.fingerprint,
                data={"answer": answer, "source": "user"},
            )
            return True, result, checkpoint, pre_submission_passed, resume_uploaded

        if action is ActionType.SAVE_CHECKPOINT:
            raw = str(decision.value or decision.args.get("checkpoint", ""))
            try:
                reached = Checkpoint(raw.strip().upper())
            except ValueError:
                result = ToolResult(
                    success=False, action=action,
                    error=f"Unknown checkpoint {raw!r}. Valid: "
                          f"{[c.value for c in Checkpoint]}",
                    url_before=state.url, url_after=state.url,
                    page_fingerprint=state.fingerprint,
                )
                return True, result, checkpoint, pre_submission_passed, resume_uploaded
            if reached not in outcome.checkpoints:
                outcome.checkpoints.append(reached)
            log.transition("Checkpoint reached: %s", reached.value)
            result = ToolResult(
                success=True, action=action, observed_state=f"Checkpoint {reached.value} saved",
                url_before=state.url, url_after=state.url, page_fingerprint=state.fingerprint,
            )
            return True, result, reached, pre_submission_passed, resume_uploaded

        if action is ActionType.VALIDATE_BEFORE_SUBMIT:
            check = validate_before_submit(
                state,
                job,
                resume_path=str(resume_path) if resume_path else "",
                resume_uploaded=resume_uploaded,
                answers=outcome.answers_used,
                fact_validator=self.fact_validator,
            )
            outcome.pre_submission = check
            if check.passed and Checkpoint.REVIEW_REACHED not in outcome.checkpoints:
                outcome.checkpoints.append(Checkpoint.REVIEW_REACHED)
            observed = (
                "Pre-submission validation PASSED. " + (
                    f"Warnings: {'; '.join(check.warnings)}" if check.warnings else ""
                )
                if check.passed
                else "Pre-submission validation FAILED: " + "; ".join(check.failures)
            )
            result = ToolResult(
                success=check.passed, action=action, observed_state=observed,
                error="" if check.passed else "; ".join(check.failures),
                url_before=state.url, url_after=state.url, page_fingerprint=state.fingerprint,
                data=check.as_dict(),
            )
            return True, result, checkpoint, check.passed, resume_uploaded

        if action is ActionType.SUBMIT_APPLICATION:
            before = state
            click = AgentDecision(
                action=ActionType.CLICK_ELEMENT,
                target=decision.target or ElementTarget(text="submit"),
                reason="Submit the application",
                expected_result="A confirmation page or message",
            )
            click_result = await self.toolbox.execute(click)
            after = await self.toolbox.context.refresh()
            evidence = validate_after_submit(before, after)
            outcome.evidence = evidence

            if evidence.confirmed:
                if Checkpoint.SUBMITTED not in outcome.checkpoints:
                    outcome.checkpoints.append(Checkpoint.SUBMITTED)
                outcome.state = ApplicationState.SUBMITTED
                log.transition("Application SUBMITTED: %s", "; ".join(evidence.signals))
            result = ToolResult(
                success=evidence.confirmed,
                action=action,
                error=(
                    ""
                    if evidence.confirmed
                    else "Submission could not be confirmed from the page. Do NOT assume it "
                         "succeeded; look for a confirmation, or report failure."
                ),
                observed_state="; ".join(evidence.signals) or click_result.observed_state,
                url_before=before.url,
                url_after=after.url,
                dom_changed=after.fingerprint != before.fingerprint,
                page_fingerprint=after.fingerprint,
                data=evidence.as_dict(),
            )
            return True, result, checkpoint, pre_submission_passed, resume_uploaded

        if action is ActionType.VALIDATE_SUBMISSION:
            after = await self.toolbox.context.refresh()
            evidence = validate_after_submit(state, after)
            outcome.evidence = evidence
            if evidence.confirmed:
                outcome.state = ApplicationState.SUBMITTED
                if Checkpoint.SUBMITTED not in outcome.checkpoints:
                    outcome.checkpoints.append(Checkpoint.SUBMITTED)
            result = ToolResult(
                success=evidence.confirmed, action=action,
                observed_state="; ".join(evidence.signals) or "No confirmation signals found",
                url_before=state.url, url_after=after.url,
                page_fingerprint=after.fingerprint, data=evidence.as_dict(),
            )
            return True, result, checkpoint, pre_submission_passed, resume_uploaded

        if action is ActionType.UPLOAD_FILE:
            # The agent decides *that* a file should be uploaded; it does not get to
            # choose *which* file. The runtime supplies the path it prepared. This
            # fixes the agent naming a file it only knows the basename of, and closes
            # the door on it nominating an arbitrary path on disk.
            if resume_path is not None:
                if decision.value and Path(decision.value) != resume_path:
                    log.info(
                        "Ignoring agent-supplied upload path %r; using the resume "
                        "prepared for this application",
                        decision.value,
                    )
                decision.value = str(resume_path)
            elif decision.value:
                return (
                    True,
                    ToolResult(
                        success=False,
                        action=action,
                        error=(
                            "No resume has been prepared for this application, and the "
                            "agent may not choose files from disk. Use request_user_input "
                            "to ask the user for the document."
                        ),
                        url_before=state.url,
                        url_after=state.url,
                        page_fingerprint=state.fingerprint,
                    ),
                    checkpoint,
                    pre_submission_passed,
                    resume_uploaded,
                )

        return False, None, checkpoint, pre_submission_passed, resume_uploaded

    # ----------------------------------------------------------------- recovery

    async def _apply_recovery(
        self,
        strategy: RecoveryStrategy,
        reason: str,
        job: JobPosting,
        outcome: ApplicationOutcome,
    ) -> str | None:
        """Turn a watchdog verdict into an instruction for the next planning turn.

        Returns ``None`` when the run must stop.
        """
        if strategy is RecoveryStrategy.RETRY:
            return "The last action failed. Retry it once, carefully."

        if strategy is RecoveryStrategy.REINSPECT_DOM:
            return (
                "Re-read the page from scratch with read_page before acting. Element ids "
                "may have changed since you last looked."
            )

        if strategy is RecoveryStrategy.REINSPECT_ACCESSIBILITY:
            snapshot = await accessibility_snapshot(self.toolbox.context.browser.page)
            lines = flatten_accessibility(snapshot, limit=80)
            return (
                "The DOM inventory was not enough. Accessibility tree:\n"
                + "\n".join(lines[:80])
            )

        if strategy is RecoveryStrategy.VISUAL_FALLBACK:
            return "VISUAL: the page appears stuck; a screenshot reading will be provided."

        if strategy is RecoveryStrategy.REPLAN:
            return (
                f"LOOP DETECTED: {reason}. Abandon your current approach entirely. "
                "Re-read the page, identify what is actually blocking progress "
                "(often an unfilled required field or a validation message), and act on that."
            )

        if strategy is RecoveryStrategy.ESCALATE_TO_USER:
            answer = await self._escalate(
                InterventionKind.NAVIGATION_FAILURE,
                job,
                f"The agent cannot make progress: {reason}",
                "Take control of the browser and either advance the form or stop this "
                "application.",
            )
            if answer is None:
                outcome.pending_user_action = UserActionRequest(
                    kind=InterventionKind.NAVIGATION_FAILURE,
                    platform=job.platform, job_title=job.role, company=job.company,
                    problem=reason,
                    action_needed="Take control of the browser, then resume or skip.",
                )
                outcome.errors.append(reason)
                return None
            self.watchdog.reset()
            return f"The user intervened and said: {answer}. Re-read the page before acting."

        outcome.errors.append(f"Abandoned: {reason}")
        return None

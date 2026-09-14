"""Application lifecycle: states, legal transitions and checkpoints.

The state machine is deliberately explicit and validated. An application can only
move along a declared edge, and every transition is persisted, so a crashed session
can be reconstructed from the transition log alone.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ApplicationState(str, Enum):
    DISCOVERED = "DISCOVERED"
    ANALYZING = "ANALYZING"
    REJECTED = "REJECTED"
    MATCHED = "MATCHED"
    OPTIMIZING_RESUME = "OPTIMIZING_RESUME"
    VALIDATING_RESUME = "VALIDATING_RESUME"
    QUEUED = "QUEUED"
    OPENING_APPLICATION = "OPENING_APPLICATION"
    APPLYING = "APPLYING"
    WAITING_USER = "WAITING_USER"
    REVIEW_PENDING = "REVIEW_PENDING"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


TERMINAL_STATES: frozenset[ApplicationState] = frozenset(
    {
        ApplicationState.REJECTED,
        ApplicationState.SUBMITTED,
        ApplicationState.FAILED,
        ApplicationState.ABANDONED,
    }
)

# Declared edges. Anything not listed here is rejected by ``StateMachine.transition``.
ALLOWED_TRANSITIONS: dict[ApplicationState, frozenset[ApplicationState]] = {
    ApplicationState.DISCOVERED: frozenset(
        {ApplicationState.ANALYZING, ApplicationState.REJECTED, ApplicationState.ABANDONED}
    ),
    ApplicationState.ANALYZING: frozenset(
        {ApplicationState.MATCHED, ApplicationState.REJECTED, ApplicationState.FAILED,
         ApplicationState.ABANDONED}
    ),
    ApplicationState.MATCHED: frozenset(
        {ApplicationState.OPTIMIZING_RESUME, ApplicationState.QUEUED, ApplicationState.ABANDONED}
    ),
    ApplicationState.OPTIMIZING_RESUME: frozenset(
        {ApplicationState.VALIDATING_RESUME, ApplicationState.FAILED, ApplicationState.ABANDONED}
    ),
    ApplicationState.VALIDATING_RESUME: frozenset(
        {ApplicationState.QUEUED, ApplicationState.OPTIMIZING_RESUME, ApplicationState.FAILED,
         ApplicationState.ABANDONED}
    ),
    ApplicationState.QUEUED: frozenset(
        {ApplicationState.OPENING_APPLICATION, ApplicationState.ABANDONED}
    ),
    ApplicationState.OPENING_APPLICATION: frozenset(
        {ApplicationState.APPLYING, ApplicationState.WAITING_USER, ApplicationState.FAILED,
         ApplicationState.ABANDONED}
    ),
    ApplicationState.APPLYING: frozenset(
        {ApplicationState.WAITING_USER, ApplicationState.REVIEW_PENDING,
         ApplicationState.SUBMITTING, ApplicationState.FAILED, ApplicationState.ABANDONED}
    ),
    ApplicationState.WAITING_USER: frozenset(
        {ApplicationState.APPLYING, ApplicationState.OPENING_APPLICATION,
         ApplicationState.ABANDONED, ApplicationState.FAILED}
    ),
    ApplicationState.REVIEW_PENDING: frozenset(
        {ApplicationState.SUBMITTING, ApplicationState.APPLYING, ApplicationState.ABANDONED}
    ),
    ApplicationState.SUBMITTING: frozenset(
        {ApplicationState.SUBMITTED, ApplicationState.APPLYING, ApplicationState.FAILED,
         ApplicationState.WAITING_USER}
    ),
    ApplicationState.SUBMITTED: frozenset(),
    ApplicationState.REJECTED: frozenset(),
    ApplicationState.FAILED: frozenset({ApplicationState.QUEUED}),
    ApplicationState.ABANDONED: frozenset({ApplicationState.QUEUED}),
}


class IllegalTransition(RuntimeError):
    def __init__(self, source: ApplicationState, target: ApplicationState) -> None:
        super().__init__(f"Illegal application transition {source.value} -> {target.value}")
        self.source = source
        self.target = target


def can_transition(source: ApplicationState, target: ApplicationState) -> bool:
    return target in ALLOWED_TRANSITIONS.get(source, frozenset())


class Checkpoint(str, Enum):
    APPLICATION_OPENED = "APPLICATION_OPENED"
    PERSONAL_INFO_COMPLETED = "PERSONAL_INFO_COMPLETED"
    EXPERIENCE_COMPLETED = "EXPERIENCE_COMPLETED"
    RESUME_UPLOADED = "RESUME_UPLOADED"
    QUESTIONNAIRE_COMPLETED = "QUESTIONNAIRE_COMPLETED"
    REVIEW_REACHED = "REVIEW_REACHED"
    SUBMITTED = "SUBMITTED"


CHECKPOINT_ORDER: tuple[Checkpoint, ...] = (
    Checkpoint.APPLICATION_OPENED,
    Checkpoint.PERSONAL_INFO_COMPLETED,
    Checkpoint.EXPERIENCE_COMPLETED,
    Checkpoint.RESUME_UPLOADED,
    Checkpoint.QUESTIONNAIRE_COMPLETED,
    Checkpoint.REVIEW_REACHED,
    Checkpoint.SUBMITTED,
)


class InterventionKind(str, Enum):
    CAPTCHA = "CAPTCHA"
    MFA = "MFA"
    SMS_CODE = "SMS_CODE"
    EMAIL_CONFIRMATION = "EMAIL_CONFIRMATION"
    PASSKEY = "PASSKEY"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"
    ACCOUNT_CREATION_APPROVAL = "ACCOUNT_CREATION_APPROVAL"
    UNKNOWN_PERSONAL_INFORMATION = "UNKNOWN_PERSONAL_INFORMATION"
    LEGAL_CONSENT = "LEGAL_CONSENT"
    SUBMIT_APPROVAL = "SUBMIT_APPROVAL"
    NAVIGATION_FAILURE = "NAVIGATION_FAILURE"
    PLATFORM_BLOCKED = "PLATFORM_BLOCKED"


class UserActionRequest(BaseModel):
    """What the UI shows in the "Action Required" queue."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    application_id: int | None = None
    kind: InterventionKind
    platform: str = ""
    job_title: str = ""
    company: str = ""
    problem: str = ""
    action_needed: str = ""
    question: str = ""
    # Free-text answer supplied by the user, if the request asks for information.
    answer: str | None = None
    resolved: bool = False


class SessionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs_discovered: int = 0
    jobs_analyzed: int = 0
    jobs_rejected: int = 0
    jobs_matched: int = 0
    applications_started: int = 0
    applications_completed: int = 0
    applications_failed: int = 0
    applications_waiting_user: int = 0
    resumes_generated: int = 0

    # Agent evaluation metrics (section 74 of the spec).
    agent_steps_total: int = 0
    tool_calls_total: int = 0
    tool_retries_total: int = 0
    vision_fallbacks: int = 0
    human_interventions: int = 0
    invalid_answers_blocked: int = 0
    execution_seconds: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return self.model_dump()


class StateMachine:
    """Validates and records application state transitions."""

    def __init__(self, state: ApplicationState = ApplicationState.DISCOVERED) -> None:
        self.state = state
        self.history: list[tuple[ApplicationState, ApplicationState, str]] = []

    def transition(self, target: ApplicationState, reason: str = "") -> ApplicationState:
        if target == self.state:
            return self.state
        if not can_transition(self.state, target):
            raise IllegalTransition(self.state, target)
        self.history.append((self.state, target, reason))
        self.state = target
        return self.state

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class ApplicationRecord(BaseModel):
    """In-memory view of an application while the agent works on it."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    job_id: int | None = None
    platform: str = ""
    external_platform: str = ""
    state: ApplicationState = ApplicationState.DISCOVERED
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    answers_used: dict[str, str] = Field(default_factory=dict)
    resume_version_id: str = ""
    decisions: list[dict] = Field(default_factory=list)

    def last_checkpoint(self) -> Checkpoint | None:
        reached = [c for c in CHECKPOINT_ORDER if c in self.checkpoints]
        return reached[-1] if reached else None

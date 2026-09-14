"""Structured agent I/O.

The agent never emits free prose that the runtime has to guess at. It emits an
``AgentDecision``, the policy engine rules on it, a tool executes it, and the tool
answers with a ``ToolResult`` the agent must then interpret.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionType(str, Enum):
    # --- perception (low risk) ---
    READ_PAGE = "read_page"
    READ_ACCESSIBILITY_TREE = "read_accessibility_tree"
    GET_URL = "get_url"
    SCROLL_PAGE = "scroll_page"
    TAKE_SCREENSHOT = "take_screenshot"
    ANALYZE_SCREENSHOT = "analyze_screenshot"
    WAIT = "wait"

    # --- navigation ---
    NAVIGATE = "navigate"
    OPEN_TAB = "open_tab"
    CLOSE_TAB = "close_tab"
    SWITCH_TAB = "switch_tab"
    GO_BACK = "go_back"

    # --- interaction (medium risk) ---
    CLICK_ELEMENT = "click_element"
    FILL_FIELD = "fill_field"
    TYPE_TEXT = "type_text"
    SELECT_OPTION = "select_option"
    SET_CHECKBOX = "set_checkbox"
    UPLOAD_FILE = "upload_file"
    PRESS_KEY = "press_key"

    # --- application lifecycle ---
    GENERATE_RESUME = "generate_resume"
    ANSWER_QUESTION = "answer_question"
    SAVE_CHECKPOINT = "save_checkpoint"
    VALIDATE_BEFORE_SUBMIT = "validate_before_submit"
    SUBMIT_APPLICATION = "submit_application"
    VALIDATE_SUBMISSION = "validate_submission"
    FINISH = "finish"
    ABANDON = "abandon"

    # --- escalation / high risk ---
    REQUEST_USER_INPUT = "request_user_input"
    LOGIN = "login"
    CREATE_ACCOUNT = "create_account"
    ACCEPT_LEGAL_TERMS = "accept_legal_terms"


#: Static risk classification. The policy engine reads this; the LLM cannot change it.
ACTION_RISK: dict[ActionType, RiskLevel] = {
    ActionType.READ_PAGE: RiskLevel.LOW,
    ActionType.READ_ACCESSIBILITY_TREE: RiskLevel.LOW,
    ActionType.GET_URL: RiskLevel.LOW,
    ActionType.SCROLL_PAGE: RiskLevel.LOW,
    ActionType.TAKE_SCREENSHOT: RiskLevel.LOW,
    ActionType.ANALYZE_SCREENSHOT: RiskLevel.LOW,
    ActionType.WAIT: RiskLevel.LOW,
    ActionType.NAVIGATE: RiskLevel.LOW,
    ActionType.OPEN_TAB: RiskLevel.LOW,
    ActionType.CLOSE_TAB: RiskLevel.LOW,
    ActionType.SWITCH_TAB: RiskLevel.LOW,
    ActionType.GO_BACK: RiskLevel.LOW,
    ActionType.SAVE_CHECKPOINT: RiskLevel.LOW,
    ActionType.VALIDATE_BEFORE_SUBMIT: RiskLevel.LOW,
    ActionType.VALIDATE_SUBMISSION: RiskLevel.LOW,
    ActionType.FINISH: RiskLevel.LOW,
    ActionType.ABANDON: RiskLevel.LOW,
    ActionType.REQUEST_USER_INPUT: RiskLevel.LOW,
    ActionType.CLICK_ELEMENT: RiskLevel.MEDIUM,
    ActionType.FILL_FIELD: RiskLevel.MEDIUM,
    ActionType.TYPE_TEXT: RiskLevel.MEDIUM,
    ActionType.SELECT_OPTION: RiskLevel.MEDIUM,
    ActionType.SET_CHECKBOX: RiskLevel.MEDIUM,
    ActionType.UPLOAD_FILE: RiskLevel.MEDIUM,
    ActionType.PRESS_KEY: RiskLevel.MEDIUM,
    ActionType.GENERATE_RESUME: RiskLevel.MEDIUM,
    ActionType.ANSWER_QUESTION: RiskLevel.MEDIUM,
    ActionType.LOGIN: RiskLevel.HIGH,
    ActionType.CREATE_ACCOUNT: RiskLevel.HIGH,
    ActionType.ACCEPT_LEGAL_TERMS: RiskLevel.HIGH,
    ActionType.SUBMIT_APPLICATION: RiskLevel.HIGH,
}


class ElementTarget(BaseModel):
    """How the agent points at something on the page.

    ``element_id`` is the stable id assigned by the perception layer (``e17``), which
    is what the agent should normally use. ``selector`` and ``text`` exist as
    recovery paths when the id has gone stale after a re-render.
    """

    model_config = ConfigDict(extra="forbid")

    element_id: str = ""
    selector: str = ""
    text: str = ""

    def is_empty(self) -> bool:
        return not (self.element_id or self.selector or self.text)


class AgentDecision(BaseModel):
    """One structured decision from the reasoning model."""

    model_config = ConfigDict(extra="forbid")

    goal: str = ""
    state: str = ""
    observation: str = ""
    action: ActionType
    target: ElementTarget = Field(default_factory=ElementTarget)
    value: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    expected_result: str = ""
    known_facts: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.5
    requires_user: bool = False

    @property
    def risk(self) -> RiskLevel:
        return ACTION_RISK.get(self.action, RiskLevel.HIGH)

    def summary(self) -> str:
        """Short, user-facing justification. Full reasoning stays internal."""
        bits = [self.action.value]
        label = self.target.element_id or self.target.text or self.target.selector
        if label:
            bits.append(f"-> {label}")
        if self.reason:
            bits.append(f"({self.reason[:120]})")
        return " ".join(bits)


class ToolResult(BaseModel):
    """Every tool answers with this. The agent must read it before deciding again."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    action: ActionType
    error: str = ""
    observed_state: str = ""
    url_before: str = ""
    url_after: str = ""
    screenshot_reference: str = ""
    dom_changed: bool = False
    page_fingerprint: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def changed_url(self) -> bool:
        return bool(self.url_after) and self.url_after != self.url_before


class PolicyOutcome(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ASK_USER = "ask_user"


class PolicyVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: PolicyOutcome
    rule: str = ""
    reason: str = ""
    intervention_kind: str = ""

    @property
    def allowed(self) -> bool:
        return self.outcome is PolicyOutcome.ALLOW


def decision_json_schema() -> dict[str, Any]:
    """JSON Schema handed to the model so it can only emit valid decisions."""
    return AgentDecision.model_json_schema()

"""Pre/post submission validators, the watchdog, and pacing."""

from __future__ import annotations

import pytest

from cvnator.agent.pacing import RateLimiter, detect_block
from cvnator.agent.validators import validate_after_submit, validate_before_submit
from cvnator.agent.watchdog import (
    RecoveryStrategy,
    Watchdog,
    cycle_fingerprint,
    strategy_for_attempt,
)
from cvnator.browser.perception import PageElement, PageState
from cvnator.domain.actions import ActionType, AgentDecision, ElementTarget, ToolResult


def make_state(
    *,
    url: str = "https://ats.example/apply/1",
    text: str = "Application form for Data Analyst at Northwind Analytics",
    elements: list[PageElement] | None = None,
    alerts: list[str] | None = None,
    fingerprint: str = "fp1",
) -> PageState:
    return PageState(
        url=url,
        title="Apply",
        text=text,
        elements=elements or [],
        alerts=alerts or [],
        lang="en",
        form_count=1,
        fingerprint=fingerprint,
    )


def element(
    element_id: str,
    *,
    tag: str = "input",
    type_: str | None = "text",
    label: str = "",
    value: str | None = "",
    required: bool = False,
    checked: bool | None = None,
) -> PageElement:
    return PageElement(
        element_id=element_id,
        tag=tag,
        type=type_,
        role=None,
        label=label,
        text="",
        name=element_id,
        placeholder=None,
        required=required,
        disabled=False,
        value=value,
        checked=checked,
        href=None,
        selector=f"#{element_id}",
    )


SUBMIT_BUTTON = element("submit", tag="button", type_="submit", label="Submit")


# ----------------------------------------------------------- pre-submission


def test_passes_a_complete_form(job):
    state = make_state(
        elements=[
            element("full_name", label="Full name", value="Ana Souza", required=True),
            SUBMIT_BUTTON,
        ]
    )
    result = validate_before_submit(state, job, resume_uploaded=False)
    assert result.passed, result.failures


def test_blocks_an_empty_required_field(job):
    state = make_state(
        elements=[
            element("full_name", label="Full name", value="", required=True),
            SUBMIT_BUTTON,
        ]
    )
    result = validate_before_submit(state, job)
    assert not result.passed
    assert any("Full name" in f for f in result.failures)


def test_blocks_when_a_validation_message_is_visible(job):
    state = make_state(
        elements=[SUBMIT_BUTTON], alerts=["Email address is not valid"]
    )
    result = validate_before_submit(state, job)
    assert not result.passed
    assert any("Email address" in f for f in result.failures)


def test_blocks_an_unticked_required_consent_box(job):
    state = make_state(
        elements=[
            element("consent", type_="checkbox", label="I consent", required=True, checked=False),
            SUBMIT_BUTTON,
        ]
    )
    result = validate_before_submit(state, job)
    assert not result.passed
    assert any("consent" in f.lower() for f in result.failures)


def test_blocks_when_a_file_input_exists_but_nothing_was_uploaded(job):
    state = make_state(
        elements=[element("resume", type_="file", label="Resume"), SUBMIT_BUTTON]
    )
    result = validate_before_submit(state, job, resume_uploaded=False)
    assert not result.passed
    assert any("file input" in f for f in result.failures)


def test_blocks_when_the_page_says_you_already_applied(job):
    state = make_state(
        text="You have already applied to this position.", elements=[SUBMIT_BUTTON]
    )
    result = validate_before_submit(state, job)
    assert not result.passed


def test_blocks_when_there_is_no_submit_control(job):
    result = validate_before_submit(make_state(elements=[]), job)
    assert not result.passed
    assert any("submit control" in f for f in result.failures)


def test_blocks_an_unsupported_answer(job, profile):
    from cvnator.resume.fact_validator import FactValidator

    state = make_state(elements=[SUBMIT_BUTTON])
    result = validate_before_submit(
        state,
        job,
        answers={"Tell us about your experience": "Expert in Kubernetes and Terraform."},
        fact_validator=FactValidator(profile),
    )
    assert not result.passed
    assert any("unsupported" in f.lower() for f in result.failures)


def test_a_missing_company_name_warns_but_does_not_block(job):
    """Plenty of application pages do not repeat the company name."""
    state = make_state(text="Application form", elements=[SUBMIT_BUTTON])
    result = validate_before_submit(state, job)
    assert result.passed
    assert result.warnings


# ---------------------------------------------------------- post-submission


@pytest.mark.parametrize(
    "text",
    [
        "Thank you for applying to this position.",
        "Your application has been submitted.",
        "Candidatura enviada com sucesso",
        "Recebemos sua candidatura",
        "Gracias por postular",
    ],
)
def test_confirmation_phrases_are_recognised(text):
    before = make_state(elements=[element("a"), element("b"), element("c")])
    after = make_state(url="https://ats.example/done", text=text, elements=[])
    assert validate_after_submit(before, after).confirmed


def test_a_reference_number_alone_confirms():
    before = make_state()
    after = make_state(text="Your application ID: AB-123456", url="https://ats.example/x")
    evidence = validate_after_submit(before, after)
    assert evidence.confirmed
    assert evidence.reference == "AB-123456"


def test_a_url_change_alone_does_not_confirm():
    """Advancing to the next wizard step is not a submission."""
    before = make_state(url="https://ats.example/apply/2", elements=[element("a")])
    after = make_state(
        url="https://ats.example/apply/3", text="Step 3 of 5", elements=[element("b")]
    )
    assert not validate_after_submit(before, after).confirmed


def test_nothing_happening_does_not_confirm():
    state = make_state()
    evidence = validate_after_submit(state, state)
    assert not evidence.confirmed
    assert evidence.signals == []


def test_already_applied_is_not_a_confirmation():
    before = make_state()
    after = make_state(
        text="You have already applied. Thank you for applying.",
        url="https://ats.example/done",
    )
    evidence = validate_after_submit(before, after)
    assert evidence.already_applied
    assert not evidence.confirmed


# ------------------------------------------------------------------ watchdog


def _result(success: bool, fingerprint: str = "fp1") -> ToolResult:
    return ToolResult(
        success=success, action=ActionType.FILL_FIELD, page_fingerprint=fingerprint
    )


def _decision(element_id: str = "e1", value: str = "x") -> AgentDecision:
    return AgentDecision(
        action=ActionType.FILL_FIELD, target=ElementTarget(element_id=element_id), value=value
    )


def test_filling_many_fields_is_not_a_stall():
    """The page fingerprint does not change when typing; that is normal, not stuck."""
    watchdog = Watchdog(stall_threshold=6)
    for index in range(8):
        verdict = watchdog.record(_decision(f"e{index}", f"value{index}"), _result(True))
        assert not verdict.stalled, f"false stall at step {index}"
        assert not verdict.looping


def test_repeating_the_same_action_is_a_loop():
    watchdog = Watchdog(loop_threshold=4)
    verdicts = [watchdog.record(_decision(), _result(False)) for _ in range(5)]
    assert any(v.looping for v in verdicts)
    assert verdicts[-1].strategy is RecoveryStrategy.REPLAN


def test_repeated_failures_on_a_frozen_page_do_stall():
    watchdog = Watchdog(stall_threshold=6, loop_threshold=99)
    verdict = None
    for index in range(6):
        verdict = watchdog.record(_decision(f"e{index}", f"v{index}"), _result(False))
    assert verdict is not None and verdict.stalled


def test_the_step_budget_is_enforced():
    watchdog = Watchdog(max_steps=3, loop_threshold=99)
    verdicts = [watchdog.record(_decision(f"e{i}", f"v{i}"), _result(True)) for i in range(3)]
    assert verdicts[-1].exhausted
    assert verdicts[-1].needs_intervention


def test_the_recovery_ladder_escalates_in_order():
    assert strategy_for_attempt(0) is RecoveryStrategy.RETRY
    assert strategy_for_attempt(3) is RecoveryStrategy.VISUAL_FALLBACK
    assert strategy_for_attempt(4) is RecoveryStrategy.REPLAN
    assert strategy_for_attempt(5) is RecoveryStrategy.ESCALATE_TO_USER
    assert strategy_for_attempt(99) is RecoveryStrategy.ABANDON


def test_cycle_fingerprint_distinguishes_target_and_value():
    base = cycle_fingerprint("page", _decision("e1", "a"))
    assert base != cycle_fingerprint("page", _decision("e2", "a"))
    assert base != cycle_fingerprint("page", _decision("e1", "b"))
    assert base != cycle_fingerprint("other", _decision("e1", "a"))


def test_reset_clears_the_watchdog():
    watchdog = Watchdog()
    watchdog.record(_decision(), _result(False))
    watchdog.reset()
    assert watchdog.steps == 0
    assert watchdog.consecutive_failures == 0


# -------------------------------------------------------------------- pacing


@pytest.mark.parametrize(
    "text",
    [
        "We've detected unusual traffic from your network",
        "Your account has been restricted",
        "Too many requests",
        "Atividade incomum detectada",
    ],
)
def test_platform_blocks_are_detected(text):
    assert detect_block(text)


def test_ordinary_pages_are_not_mistaken_for_blocks():
    assert not detect_block("Apply for this position. We review every application.")


def test_daily_limit_tracking():
    limiter = RateLimiter()
    limiter.default.daily_limit = 2
    assert not limiter.daily_limit_reached("mock")
    limiter.record_application("mock")
    limiter.record_application("mock")
    assert limiter.daily_limit_reached("mock")
    assert limiter.applications_today("mock") == 2
    assert not limiter.daily_limit_reached("other")


def test_block_detection_ignores_markup_and_scripts():
    """Run against visible text, not raw HTML.

    "rate limit" appears in plenty of JavaScript bundles and CSS class names. Reading
    the raw document would pause the whole session on a page that is working fine.
    """
    # A bundle carrying the strings for an error state it is not currently showing.
    html = (
        "<html><head><script>"
        'const MESSAGES={429:"Too many requests, please slow down",'
        '403:"Access denied"};'
        "</script></head>"
        "<body><h1>Data Analyst</h1><p>Apply for this position.</p></body></html>"
    )
    assert detect_block(html), "raw HTML does trip the detector - which is why we do not use it"

    visible_text = "Data Analyst\nApply for this position."
    assert not detect_block(visible_text)


def test_a_genuine_block_message_is_still_caught():
    assert detect_block("We have detected unusual traffic from your network.")
    assert detect_block("Your account has been restricted.")

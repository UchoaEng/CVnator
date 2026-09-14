"""The policy engine must not be arguable with. These lock its rulings down."""

from __future__ import annotations

import pytest

from cvnator.agent.policy import PolicyContext, PolicyEngine, detect_challenge
from cvnator.domain.actions import ActionType, AgentDecision, ElementTarget, PolicyOutcome
from cvnator.domain.search import AutonomySettings


@pytest.fixture
def engine(profile):
    return PolicyEngine(profile)


def ctx(profile, **overrides):
    defaults = dict(
        autonomy=AutonomySettings(dry_run=False, auto_submit=True, review_before_submit=False),
        profile=profile,
    )
    defaults.update(overrides)
    return PolicyContext(**defaults)


@pytest.mark.parametrize(
    "text",
    [
        "Please complete the reCAPTCHA to continue",
        "Não sou um robô",
        "Enter the verification code we sent by SMS",
        "Two-factor authentication required",
    ],
)
def test_challenges_are_detected(text):
    assert detect_challenge(text)


def test_never_interacts_with_a_captcha(engine, profile):
    decision = AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(element_id="e1"))
    verdict = engine.evaluate(
        decision, ctx(profile, page_text="Please complete the reCAPTCHA to continue")
    )
    assert verdict.outcome is PolicyOutcome.ASK_USER
    assert verdict.rule == "no_challenge_bypass"


def test_blocks_a_fabricated_skill_before_it_is_typed(engine, profile):
    decision = AgentDecision(
        action=ActionType.FILL_FIELD,
        target=ElementTarget(element_id="e5"),
        value="I have extensive production experience with Kubernetes and Terraform.",
    )
    verdict = engine.evaluate(decision, ctx(profile, field_label="Tell us about your experience"))
    assert verdict.outcome is PolicyOutcome.BLOCK
    assert verdict.rule == "fact_integrity"
    # The agent must be told not to reword it, or it will simply try again.
    assert "rephrase" in verdict.reason.lower()


def test_blocks_an_inflated_number_in_a_years_field(engine, profile):
    """The value is a bare "8"; only the field label reveals it is a claim."""
    decision = AgentDecision(
        action=ActionType.FILL_FIELD, target=ElementTarget(element_id="e7"), value="8"
    )
    verdict = engine.evaluate(
        decision, ctx(profile, field_label="How many years of experience do you have with Python?")
    )
    assert verdict.outcome is PolicyOutcome.BLOCK
    assert "python" in verdict.reason.lower()


def test_allows_a_supported_number_in_a_years_field(engine, profile):
    decision = AgentDecision(
        action=ActionType.FILL_FIELD, target=ElementTarget(element_id="e7"), value="3"
    )
    verdict = engine.evaluate(
        decision, ctx(profile, field_label="How many years of experience do you have with Python?")
    )
    assert verdict.outcome is PolicyOutcome.ALLOW


def test_dry_run_blocks_submission_but_not_form_filling(engine, profile):
    autonomy = AutonomySettings(dry_run=True, auto_submit=True, review_before_submit=False)
    submit = AgentDecision(action=ActionType.SUBMIT_APPLICATION)
    verdict = engine.evaluate(
        submit, ctx(profile, autonomy=autonomy, pre_submission_passed=True)
    )
    assert verdict.outcome is PolicyOutcome.BLOCK
    assert verdict.rule == "dry_run"

    fill = AgentDecision(
        action=ActionType.FILL_FIELD, target=ElementTarget(element_id="e1"), value="Ana Souza"
    )
    assert engine.evaluate(fill, ctx(profile, autonomy=autonomy)).outcome is PolicyOutcome.ALLOW


def test_submission_requires_pre_validation(engine, profile):
    verdict = engine.evaluate(
        AgentDecision(action=ActionType.SUBMIT_APPLICATION),
        ctx(profile, pre_submission_passed=False),
    )
    assert verdict.outcome is PolicyOutcome.BLOCK
    assert verdict.rule == "submission_requires_validation"


def test_review_before_submit_escalates_instead_of_sending(engine, profile):
    autonomy = AutonomySettings(dry_run=False, auto_submit=True, review_before_submit=True)
    verdict = engine.evaluate(
        AgentDecision(action=ActionType.SUBMIT_APPLICATION),
        ctx(profile, autonomy=autonomy, pre_submission_passed=True),
    )
    assert verdict.outcome is PolicyOutcome.ASK_USER


def test_auto_submit_allowed_only_when_everything_lines_up(engine, profile):
    autonomy = AutonomySettings(dry_run=False, auto_submit=True, review_before_submit=False)
    verdict = engine.evaluate(
        AgentDecision(action=ActionType.SUBMIT_APPLICATION),
        ctx(profile, autonomy=autonomy, pre_submission_passed=True),
    )
    assert verdict.outcome is PolicyOutcome.ALLOW


def test_daily_limit_blocks_further_submissions(engine, profile):
    autonomy = AutonomySettings(
        dry_run=False, auto_submit=True, review_before_submit=False, daily_application_limit=5
    )
    verdict = engine.evaluate(
        AgentDecision(action=ActionType.SUBMIT_APPLICATION),
        ctx(profile, autonomy=autonomy, pre_submission_passed=True, applications_today=5),
    )
    assert verdict.outcome is PolicyOutcome.BLOCK
    assert verdict.rule == "daily_limit"


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("never", PolicyOutcome.BLOCK),
        ("ask", PolicyOutcome.ASK_USER),
        ("auto", PolicyOutcome.ALLOW),
    ],
)
def test_account_creation_follows_the_configured_policy(engine, profile, policy, expected):
    autonomy = AutonomySettings(
        dry_run=False, account_policy=policy, auto_create_external_account=(policy == "auto")
    )
    verdict = engine.evaluate(
        AgentDecision(action=ActionType.CREATE_ACCOUNT), ctx(profile, autonomy=autonomy)
    )
    assert verdict.outcome is expected


def test_legal_consent_is_always_the_users_decision(engine, profile):
    verdict = engine.evaluate(
        AgentDecision(action=ActionType.ACCEPT_LEGAL_TERMS),
        ctx(profile, autonomy=AutonomySettings(dry_run=False, auto_submit=True)),
    )
    assert verdict.outcome is PolicyOutcome.ASK_USER


def test_login_without_stored_credentials_escalates(engine, profile):
    verdict = engine.evaluate(
        AgentDecision(action=ActionType.LOGIN),
        ctx(profile, has_stored_credentials=False, platform="gupy"),
    )
    assert verdict.outcome is PolicyOutcome.ASK_USER


def test_reading_the_page_is_never_gated(engine, profile):
    verdict = engine.evaluate(AgentDecision(action=ActionType.READ_PAGE), ctx(profile))
    assert verdict.outcome is PolicyOutcome.ALLOW


def test_a_years_field_is_judged_against_that_specific_skill(profile):
    """"6" in a SQL field is a SQL claim, not a claim about total experience.

    A greedy subject match turns "years of experience do you have with SQL" into a
    phrase matching no skill, which silently falls back to total experience - and
    lets someone with 3 years of SQL answer 6.
    """
    from cvnator.domain.profile import Skill

    profile.skills = [
        Skill(name="Python", category="hard", years_of_experience=4.0),
        Skill(name="SQL", category="hard", years_of_experience=3.0),
    ]
    engine = PolicyEngine(profile)
    label = "How many years of experience do you have with SQL?"

    blocked = engine.evaluate(
        AgentDecision(
            action=ActionType.FILL_FIELD, target=ElementTarget(element_id="e0"), value="6"
        ),
        ctx(profile, field_label=label),
    )
    assert blocked.outcome is PolicyOutcome.BLOCK
    assert "sql" in blocked.reason.lower()
    # The agent must be told what to use, or it escalates instead of self-correcting.
    assert "Enter 3 instead" in blocked.reason

    allowed = engine.evaluate(
        AgentDecision(
            action=ActionType.FILL_FIELD, target=ElementTarget(element_id="e0"), value="3"
        ),
        ctx(profile, field_label=label),
    )
    assert allowed.outcome is PolicyOutcome.ALLOW


def test_a_generic_years_field_uses_total_experience(profile):
    engine = PolicyEngine(profile)
    total = profile.total_years_experience()
    verdict = engine.evaluate(
        AgentDecision(
            action=ActionType.FILL_FIELD,
            target=ElementTarget(element_id="e0"),
            value=str(int(total)),
        ),
        ctx(profile, field_label="Total years of professional experience"),
    )
    assert verdict.outcome is PolicyOutcome.ALLOW

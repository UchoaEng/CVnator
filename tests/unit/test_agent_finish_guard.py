"""What happens when the agent claims to be done and the page disagrees.

The guard that refuses an unverified `finish` is correct and must stay. But a
refusal the agent cannot act on costs the whole application: in the run that
prompted these tests the agent clicked *next* on the last form step, landed on
the review screen, was told only that the submission "could not be confirmed",
and gave up one control short of the end.

So the guard now sends the agent back with the reason and a move -- a bounded
number of times. These tests pin all three halves of that: it refuses, it
recovers, and it still terminates.

The last test covers the same shape of mistake one layer up: an answer the model
truncated mid-JSON used to end the application outright.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from cvnator.agent.loop import MAX_FINISH_REJECTIONS, MAX_PLANNING_FAILURES, FormAgent
from cvnator.agent.policy import PolicyEngine
from cvnator.browser.perception import PageElement, PageState
from cvnator.domain.actions import ActionType, AgentDecision, ElementTarget
from cvnator.domain.application import ApplicationState
from cvnator.domain.job import JobPosting
from cvnator.domain.search import AutonomySettings

REVIEW_PAGE_TEXT = """\
Review your application
Please check your answers before finishing.
Name: Lucas Ismael
Email: lucas.ismael@example.com
"""

CONFIRMATION_PAGE_TEXT = """\
Application submitted
Thank you for applying.
Your reference number is MA-778899.
"""


def _button(element_id: str, text: str) -> PageElement:
    return PageElement(
        element_id=element_id,
        tag="button",
        type="submit",
        role="button",
        label=text,
        text=text,
        name="action",
        placeholder=None,
        required=False,
        disabled=False,
        value=None,
        checked=None,
        href=None,
        selector=f"#{element_id}",
    )


def _page(url: str, text: str, fingerprint: str) -> PageState:
    return PageState(
        url=url,
        title="Application",
        text=text,
        elements=[_button("e0", "Submit application")],
        alerts=[],
        lang="en",
        form_count=1,
        fingerprint=fingerprint,
    )


REVIEW_PAGE = _page("http://127.0.0.1:8099/apply/tok/5", REVIEW_PAGE_TEXT, "fp-review")
CONFIRMED_PAGE = _page("http://127.0.0.1:8099/apply/tok/done", CONFIRMATION_PAGE_TEXT, "fp-done")


class _ScriptedGateway:
    """Returns pre-written decisions and records the context it was asked about."""

    def __init__(self, decisions: Sequence[AgentDecision | Exception]) -> None:
        self._decisions = list(decisions)
        self.contexts: list[str] = []

    async def structured(self, _role, _model, messages, **_kwargs) -> AgentDecision:
        self.contexts.append(messages[-1].content)
        if not self._decisions:
            raise AssertionError("the agent asked for more decisions than the test scripted")
        scripted = self._decisions.pop(0)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted


class _ScriptedContext:
    def __init__(self, pages: list[PageState]) -> None:
        self._pages = pages

    async def refresh(self) -> PageState:
        # The last page persists, so an over-eager agent keeps seeing the same thing
        # rather than running off the end of the script.
        return self._pages.pop(0) if len(self._pages) > 1 else self._pages[0]


class _ScriptedToolbox:
    def __init__(self, pages: list[PageState]) -> None:
        self.context = _ScriptedContext(pages)
        self.executed: list[ActionType] = []

    async def execute(self, decision: AgentDecision):
        self.executed.append(decision.action)
        raise AssertionError("no tool should run in these tests")


def _agent(gateway: _ScriptedGateway, pages: list[PageState], db_session) -> FormAgent:
    from cvnator.agent.memory import MemoryStore

    return FormAgent(
        gateway,  # type: ignore[arg-type]
        _ScriptedToolbox(pages),  # type: ignore[arg-type]
        PolicyEngine(_PROFILE),
        MemoryStore(db_session),
        _PROFILE,
        AutonomySettings(dry_run=False),
        max_steps=20,
    )


def _finish(reason: str = "The form is gone, so the application must be submitted.") -> AgentDecision:
    return AgentDecision(
        action=ActionType.FINISH,
        target=ElementTarget(),
        reason=reason,
        expected_result="The application is complete",
    )


_JOB = JobPosting(
    platform="mock",
    platform_job_id="MJ-1001",
    company="Northwind Analytics",
    role="Data Analyst",
    url="http://127.0.0.1:8099/jobs/MJ-1001",
)


@pytest.fixture(autouse=True)
def _profile_fixture(profile):
    global _PROFILE
    _PROFILE = profile
    yield


@pytest.mark.asyncio
async def test_finish_on_an_unconfirmed_page_sends_the_agent_back_with_a_move(
    db_session,
) -> None:
    gateway = _ScriptedGateway([_finish(), _finish()])
    agent = _agent(gateway, [REVIEW_PAGE], db_session)

    outcome = await agent.run(_JOB, resume_path=Path("resume.pdf"))

    # Refused, so not submitted -- and the agent was told why and what to do.
    assert outcome.state is not ApplicationState.SUBMITTED
    assert len(gateway.contexts) >= 2
    second_prompt = gateway.contexts[1]
    assert "NOT submitted" in second_prompt
    assert "finalises the application" in second_prompt


@pytest.mark.asyncio
async def test_the_agent_can_still_reach_submitted_after_being_sent_back(
    db_session,
) -> None:
    """The point of the retry: the page confirms on the second look."""
    gateway = _ScriptedGateway([_finish(), _finish()])
    agent = _agent(gateway, [REVIEW_PAGE, CONFIRMED_PAGE], db_session)

    outcome = await agent.run(_JOB, resume_path=Path("resume.pdf"))

    assert outcome.state is ApplicationState.SUBMITTED
    assert outcome.evidence is not None and outcome.evidence.confirmed
    assert any("MA-778899" in signal for signal in outcome.evidence.signals)


@pytest.mark.asyncio
async def test_a_repeated_unverified_finish_terminates_as_a_failure(db_session) -> None:
    """Sending the agent back is bounded: a confused agent must not loop forever."""
    gateway = _ScriptedGateway([_finish() for _ in range(MAX_FINISH_REJECTIONS + 5)])
    agent = _agent(gateway, [REVIEW_PAGE], db_session)

    outcome = await agent.run(_JOB, resume_path=Path("resume.pdf"))

    assert outcome.state is ApplicationState.FAILED
    assert len(gateway.contexts) == MAX_FINISH_REJECTIONS + 1
    assert any("could not be verified" in error for error in outcome.errors)


@pytest.mark.asyncio
async def test_an_unreadable_model_answer_is_retried_not_fatal(db_session) -> None:
    """A truncated answer is a bad turn, not a bad application.

    The observed failure: the model narrated its way past the token budget while
    describing the review page, the JSON came back cut off, and an application that
    was one click from done was recorded as FAILED.
    """
    truncated = ValueError('No JSON object found in response: \'{"goal": "Complete')
    gateway = _ScriptedGateway([truncated, _finish()])
    agent = _agent(gateway, [REVIEW_PAGE, CONFIRMED_PAGE], db_session)

    outcome = await agent.run(_JOB, resume_path=Path("resume.pdf"))

    assert outcome.state is ApplicationState.SUBMITTED
    assert "cut off" in gateway.contexts[1]


@pytest.mark.asyncio
async def test_unreadable_answers_still_end_the_application_eventually(db_session) -> None:
    gateway = _ScriptedGateway(
        [ValueError("No JSON object found in response") for _ in range(MAX_PLANNING_FAILURES + 3)]
    )
    agent = _agent(gateway, [REVIEW_PAGE], db_session)

    outcome = await agent.run(_JOB, resume_path=Path("resume.pdf"))

    assert outcome.state is ApplicationState.FAILED
    assert any("Planning failed" in error for error in outcome.errors)
    assert len(gateway.contexts) == MAX_PLANNING_FAILURES + 1

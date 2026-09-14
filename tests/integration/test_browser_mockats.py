"""Browser stack against the real MockATS, with no model in the loop.

These tests drive perception and the tool layer directly with scripted decisions.
That isolates the deterministic half of the agent — can it see the page correctly, does
a tool report honestly — from the model's judgement, so a regression in either is
attributable.

Skipped automatically when Chromium is not installed.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import pytest
import uvicorn

from cvnator.browser.controller import BrowserController
from cvnator.browser.perception import compute_fingerprint, needs_visual_fallback, observe
from cvnator.browser.tools import BrowserToolbox, ToolContext, resolve_target
from cvnator.config import get_settings
from cvnator.domain.actions import ActionType, AgentDecision, ElementTarget

pytestmark = pytest.mark.asyncio


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def mock_ats() -> Iterator[str]:
    from cvnator.mockats.server import create_app

    port = _free_port()
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(100):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                break
        threading.Event().wait(0.05)
    else:  # pragma: no cover - the harness itself failed to start
        pytest.skip("MockATS did not start")

    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
async def browser() -> Iterator[BrowserController]:
    controller = BrowserController(get_settings(), headless=True)
    try:
        await controller.start()
    except Exception as exc:  # pragma: no cover - environment without Chromium
        pytest.skip(f"Chromium unavailable: {exc}")
    yield controller
    await controller.stop()


async def _goto(browser: BrowserController, url: str):
    result = await browser.navigate(url)
    assert result.ok, result.error
    return await observe(browser.page)


async def test_perception_labels_every_required_field(browser, mock_ats):
    """Including the input whose <label> is deliberately not associated via `for`."""
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    state = await observe(browser.page)

    labels = {e.name: e.label for e in state.elements if e.name}
    assert "full_name" in labels
    # `city` has no `for=` association; the label must still be recovered from layout.
    assert "city" in labels
    assert "City of residence" in labels["city"]

    # Required-ness comes from two places, and both must be detected: the HTML
    # attribute, and a "*" in the label with server-side validation only.
    required = {e.name: e.required_source for e in state.elements if e.required}
    assert required.get("full_name") == "attribute"
    assert required.get("city") == "label", "a starred label must count as required"
    assert {"full_name", "email", "city"} <= set(required)


async def test_generated_selectors_are_unique(browser, mock_ats):
    """Two submit buttons share name="action"; an ambiguous selector clicks the wrong one."""
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    # Step 1 has no Back button; step 2 has both.
    await _fill_step_one(toolbox)

    state = await observe(browser.page)
    buttons = [e for e in state.elements if e.tag == "button"]
    assert len(buttons) >= 2
    for button in buttons:
        matches = await browser.page.locator(button.selector).count()
        assert matches == 1, f"{button.selector!r} matched {matches} elements"


async def _fill_step_one(toolbox: BrowserToolbox) -> None:
    for name, value in (
        ("full_name", "Ana Souza"),
        ("email", "ana@example.com"),
        ("phone", "+55 11 90000-0000"),
        ("city", "São Paulo"),
    ):
        state = await toolbox.context.refresh()
        element = next(e for e in state.elements if e.name == name)
        result = await toolbox.execute(
            AgentDecision(
                action=ActionType.FILL_FIELD,
                target=ElementTarget(element_id=element.element_id),
                value=value,
            )
        )
        assert result.success, result.error

    state = await toolbox.context.refresh()
    continue_button = next(e for e in state.elements if e.element_id and "continue" in e.text.lower())
    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.CLICK_ELEMENT,
            target=ElementTarget(element_id=continue_button.element_id),
        )
    )
    assert result.success, result.error


async def test_fill_reports_the_value_that_actually_landed(browser, mock_ats):
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    state = await toolbox.context.refresh()
    element = next(e for e in state.elements if e.name == "full_name")

    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.FILL_FIELD,
            target=ElementTarget(element_id=element.element_id),
            value="Ana Souza",
        )
    )
    assert result.success
    assert result.data["value_after"] == "Ana Souza"


async def test_a_click_that_changes_nothing_is_reported_as_failure(browser, mock_ats):
    """Submitting an empty required form must not be recorded as a success."""
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    state = await toolbox.context.refresh()
    continue_button = next(e for e in state.elements if "continue" in e.text.lower())

    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.CLICK_ELEMENT,
            target=ElementTarget(element_id=continue_button.element_id),
        )
    )
    # Which layer rejects it depends on the field: the browser blocks the submit for
    # attribute-required fields, the server re-renders for star-only ones. Either way
    # the reason must be visible to the agent, and the click must not read as success.
    state = await toolbox.context.refresh()
    assert state.alerts, "validation errors must be surfaced to the agent"
    assert any(
        term in alert.lower()
        for alert in state.alerts
        for term in ("required", "fill out", "preencha", "obrigat")
    ), state.alerts
    if not result.success:
        assert result.error


async def test_html5_constraint_violations_are_surfaced(browser, mock_ats):
    """A native validation bubble never reaches the DOM; the agent must still see it."""
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _fill_step_one(toolbox)

    state = await toolbox.context.refresh()
    years = next(e for e in state.elements if e.name == "years_experience")
    assert years.constraints is not None
    assert years.constraints["step"] == "0.5"

    await toolbox.execute(
        AgentDecision(
            action=ActionType.FILL_FIELD,
            target=ElementTarget(element_id=years.element_id),
            value="6.2",
        )
    )
    state = await toolbox.context.refresh()
    invalid = next(e for e in state.elements if e.name == "years_experience")
    assert invalid.invalid
    assert invalid.validation_message
    assert any("years_experience" in a or "experience" in a.lower() for a in state.alerts)


async def test_select_option_matches_by_label(browser, mock_ats):
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _fill_step_one(toolbox)

    state = await toolbox.context.refresh()
    seniority = next(e for e in state.elements if e.name == "seniority")
    assert [o["value"] for o in seniority.options][:3] == ["", "intern", "junior"]

    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.SELECT_OPTION,
            target=ElementTarget(element_id=seniority.element_id),
            value="Mid",
        )
    )
    assert result.success, result.error


async def test_fingerprint_is_stable_across_reloads_but_changes_between_pages(browser, mock_ats):
    first = await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    again = await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    other = await _goto(browser, f"{mock_ats}/jobs/MJ-1002")

    assert first.fingerprint == again.fingerprint
    assert first.fingerprint != other.fingerprint


async def test_ordinary_form_does_not_request_vision(browser, mock_ats):
    """The expensive path must not fire on a page the DOM describes perfectly well."""
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    state = await toolbox.context.refresh()

    needed, reason = needs_visual_fallback(state)
    assert not needed, f"vision requested unnecessarily: {reason}"


async def test_target_resolution_falls_back_from_id_to_text(browser, mock_ats):
    state = await _goto(browser, f"{mock_ats}/jobs/MJ-1001")

    by_text = resolve_target(
        state, AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    assert by_text.strategy == "text_match"
    assert by_text.selector

    unresolved = resolve_target(
        state,
        AgentDecision(
            action=ActionType.CLICK_ELEMENT, target=ElementTarget(element_id="e9999")
        ),
    )
    assert unresolved.strategy == "unresolved"


async def test_accessibility_tree_is_available_via_cdp(browser, mock_ats):
    """Playwright removed page.accessibility, so this must come from CDP."""
    from cvnator.browser.perception import accessibility_snapshot, flatten_accessibility

    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    snapshot = await accessibility_snapshot(browser.page)
    lines = flatten_accessibility(snapshot)
    assert lines, "accessibility snapshot was empty"
    assert any("button" in line.lower() or "apply" in line.lower() for line in lines)


async def test_upload_is_verified_against_the_page(browser, mock_ats, sample_pdf):
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _fill_step_one(toolbox)

    # Complete step 2 so the form advances to the upload step.
    for name, value in (("current_title", "Data Analyst"), ("current_company", "Acme Log")):
        state = await toolbox.context.refresh()
        element = next(e for e in state.elements if e.name == name)
        await toolbox.execute(
            AgentDecision(
                action=ActionType.FILL_FIELD,
                target=ElementTarget(element_id=element.element_id),
                value=value,
            )
        )
    state = await toolbox.context.refresh()
    years = next(e for e in state.elements if e.name == "years_experience")
    await toolbox.execute(
        AgentDecision(
            action=ActionType.FILL_FIELD,
            target=ElementTarget(element_id=years.element_id),
            value="6",
        )
    )
    state = await toolbox.context.refresh()
    seniority = next(e for e in state.elements if e.name == "seniority")
    await toolbox.execute(
        AgentDecision(
            action=ActionType.SELECT_OPTION,
            target=ElementTarget(element_id=seniority.element_id),
            value="mid",
        )
    )
    state = await toolbox.context.refresh()
    nxt = next(e for e in state.elements if e.tag == "button" and "continue" in e.text.lower())
    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.CLICK_ELEMENT, target=ElementTarget(element_id=nxt.element_id)
        )
    )
    assert result.success, result.error

    state = await toolbox.context.refresh()
    file_input = next(e for e in state.elements if e.type == "file")
    upload = await toolbox.execute(
        AgentDecision(
            action=ActionType.UPLOAD_FILE,
            target=ElementTarget(element_id=file_input.element_id),
            value=str(sample_pdf),
        )
    )
    assert upload.success, upload.error
    assert upload.data["file"] == sample_pdf.name


async def test_fingerprint_ignores_volatile_numbers():
    """Two renders differing only in a timestamp are the same page, not progress."""
    elements = [{"tag": "input", "type": "text", "role": None, "label": "Name"}]
    first = compute_fingerprint("https://x/a", "T", elements, "Generated at 12:04:11")
    second = compute_fingerprint("https://x/a", "T", elements, "Generated at 12:09:57")
    assert first == second


async def test_form_progress_lists_what_is_left(browser, mock_ats):
    """The agent gets an explicit checklist, not just 20 lines of attributes."""
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )

    state = await toolbox.context.refresh()
    empty = state.form_progress()
    assert "0 of" in empty
    assert "STILL EMPTY" in empty
    assert "ALREADY FILLED" not in empty

    element = next(e for e in state.elements if e.name == "full_name")
    await toolbox.execute(
        AgentDecision(
            action=ActionType.FILL_FIELD,
            target=ElementTarget(element_id=element.element_id),
            value="Ana Souza",
        )
    )
    state = await toolbox.context.refresh()
    partial = state.form_progress()
    assert "1 of" in partial
    assert "do not refill" in partial
    assert "Full name" in partial


async def _reach_questions_step(toolbox: BrowserToolbox, sample_pdf) -> None:
    """Drive the wizard as far as the screening questions."""
    await _fill_step_one(toolbox)
    for name, value in (
        ("current_title", "Data Analyst"),
        ("current_company", "Acme Log"),
        ("years_experience", "6"),
    ):
        state = await toolbox.context.refresh()
        el = next(e for e in state.elements if e.name == name)
        await toolbox.execute(
            AgentDecision(
                action=ActionType.FILL_FIELD,
                target=ElementTarget(element_id=el.element_id),
                value=value,
            )
        )
    state = await toolbox.context.refresh()
    seniority = next(e for e in state.elements if e.name == "seniority")
    await toolbox.execute(
        AgentDecision(
            action=ActionType.SELECT_OPTION,
            target=ElementTarget(element_id=seniority.element_id),
            value="mid",
        )
    )
    state = await toolbox.context.refresh()
    nxt = next(e for e in state.elements if e.tag == "button" and "continue" in e.text.lower())
    await toolbox.execute(
        AgentDecision(
            action=ActionType.CLICK_ELEMENT, target=ElementTarget(element_id=nxt.element_id)
        )
    )
    state = await toolbox.context.refresh()
    file_input = next(e for e in state.elements if e.type == "file")
    await toolbox.execute(
        AgentDecision(
            action=ActionType.UPLOAD_FILE,
            target=ElementTarget(element_id=file_input.element_id),
            value=str(sample_pdf),
        )
    )
    state = await toolbox.context.refresh()
    nxt = next(e for e in state.elements if e.tag == "button" and "continue" in e.text.lower())
    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.CLICK_ELEMENT, target=ElementTarget(element_id=nxt.element_id)
        )
    )
    assert result.success, result.error


async def test_clicking_a_checkbox_is_verified_by_its_state(browser, mock_ats, sample_pdf):
    """Toggling a checkbox leaves the page identical; that is success, not failure."""
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _reach_questions_step(toolbox, sample_pdf)

    state = await toolbox.context.refresh()
    consent = next(e for e in state.elements if e.name == "consent")
    assert consent.checked is False

    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.CLICK_ELEMENT,
            target=ElementTarget(element_id=consent.element_id),
        )
    )
    assert result.success, result.error
    assert result.data["checked_before"] is False
    assert result.data["checked_after"] is True

    state = await toolbox.context.refresh()
    assert next(e for e in state.elements if e.name == "consent").checked is True


async def test_set_checkbox_also_works_on_a_radio_group(browser, mock_ats, sample_pdf):
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _reach_questions_step(toolbox, sample_pdf)

    state = await toolbox.context.refresh()
    # Both radios share name="work_auth"; the selector must still target one of them.
    yes_radio = next(
        e for e in state.elements if e.name == "work_auth" and e.type == "radio"
    )
    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.SET_CHECKBOX,
            target=ElementTarget(element_id=yes_radio.element_id),
            args={"checked": True},
        )
    )
    assert result.success, result.error


async def test_a_long_answer_is_reported_as_clipped_not_truncated(browser, mock_ats, sample_pdf):
    """The display is clipped for transport; the agent must not read that as data loss.

    Showing a clipped value with no marker reads as "the field content was truncated",
    and the agent dutifully rewrites an answer that was perfectly fine.
    """
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _reach_questions_step(toolbox, sample_pdf)

    long_answer = (
        "I have spent the last several years building operational dashboards and "
        "modelling data in SQL, which is exactly what this role describes. "
    ) * 3
    state = await toolbox.context.refresh()
    motivation = next(e for e in state.elements if e.name == "motivation")
    await toolbox.execute(
        AgentDecision(
            action=ActionType.FILL_FIELD,
            target=ElementTarget(element_id=motivation.element_id),
            value=long_answer,
        )
    )

    state = await toolbox.context.refresh()
    updated = next(e for e in state.elements if e.name == "motivation")
    assert updated.value_length == len(long_answer)
    description = updated.describe()
    assert "chars total, shown clipped" in description
    assert str(len(long_answer)) in description


async def test_clicking_an_already_selected_radio_is_not_a_failure(
    browser, mock_ats, sample_pdf
):
    """A radio cannot be un-selected by clicking it.

    Treating "state unchanged" as failure made the agent conclude the control was
    broken, click the other option to "unstick" it, then click back — three failures
    in a row on a form that was doing exactly what it should.
    """
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _reach_questions_step(toolbox, sample_pdf)

    state = await toolbox.context.refresh()
    yes_radio = next(
        e for e in state.elements if e.name == "work_auth" and (e.value or "") == "yes"
    )

    first = await toolbox.execute(
        AgentDecision(
            action=ActionType.CLICK_ELEMENT,
            target=ElementTarget(element_id=yes_radio.element_id),
        )
    )
    assert first.success, first.error

    state = await toolbox.context.refresh()
    yes_radio = next(
        e for e in state.elements if e.name == "work_auth" and (e.value or "") == "yes"
    )
    assert yes_radio.checked is True

    # Clicking it again is a no-op, and must still read as success.
    second = await toolbox.execute(
        AgentDecision(
            action=ActionType.CLICK_ELEMENT,
            target=ElementTarget(element_id=yes_radio.element_id),
        )
    )
    assert second.success, second.error
    assert second.data["checked_after"] is True


async def test_a_radio_group_counts_as_one_requirement(browser, mock_ats, sample_pdf):
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _reach_questions_step(toolbox, sample_pdf)

    state = await toolbox.context.refresh()
    before = state.form_progress()
    assert "choose one" in before, before

    yes_radio = next(
        e for e in state.elements if e.name == "work_auth" and (e.value or "") == "yes"
    )
    await toolbox.execute(
        AgentDecision(
            action=ActionType.SET_CHECKBOX,
            target=ElementTarget(element_id=yes_radio.element_id),
            args={"checked": True},
        )
    )

    state = await toolbox.context.refresh()
    after = state.form_progress()
    assert "choose one" not in after, after
    # The group moved from outstanding to done as a single unit.
    assert "authorized to work" in after.split("ALREADY FILLED")[-1].lower()


async def test_filling_a_long_answer_reports_success(browser, mock_ats, sample_pdf):
    """A multi-sentence answer must not read as a failed fill.

    The echoed value is clipped at 200 characters for transport, so comparing it to
    what was typed always mismatched — and the agent dutifully rewrote text that had
    landed perfectly.
    """
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _reach_questions_step(toolbox, sample_pdf)

    long_answer = (
        "I have spent several years building operational dashboards and modelling "
        "data in SQL, which is what this role describes. I would like to keep doing "
        "that work somewhere it is central rather than incidental."
    )
    assert len(long_answer) > 200, "the point of the test is to exceed the transport clip"

    state = await toolbox.context.refresh()
    motivation = next(e for e in state.elements if e.name == "motivation")
    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.FILL_FIELD,
            target=ElementTarget(element_id=motivation.element_id),
            value=long_answer,
        )
    )
    assert result.success, result.error
    assert result.data["value_length"] == len(long_answer)


async def test_a_genuinely_rejected_fill_still_reports_failure(browser, mock_ats):
    """The relaxed comparison must not paper over a value the field refused."""
    await _goto(browser, f"{mock_ats}/jobs/MJ-1001")
    toolbox = BrowserToolbox(ToolContext(browser))
    await toolbox.execute(
        AgentDecision(action=ActionType.CLICK_ELEMENT, target=ElementTarget(text="Apply"))
    )
    await _fill_step_one(toolbox)

    # maxlength is not set here, so use a number input given non-numeric text: the
    # browser discards it and the field stays empty.
    state = await toolbox.context.refresh()
    years = next(e for e in state.elements if e.name == "years_experience")
    result = await toolbox.execute(
        AgentDecision(
            action=ActionType.FILL_FIELD,
            target=ElementTarget(element_id=years.element_id),
            value="six point two",
        )
    )
    assert not result.success
    assert result.error

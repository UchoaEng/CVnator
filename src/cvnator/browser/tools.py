"""The tool layer the agent acts through.

Two ideas carry most of the weight here:

**Nothing is assumed to have worked.** Every tool re-observes the page after acting
and reports what actually changed - the URL, the page fingerprint, the field's real
value, any validation message that appeared. A click that Playwright performed
without raising is still a failure if the page did not move.

**Element ids are resolved with fallbacks.** The agent points at ``e17``; by the time
the click lands the page may have re-rendered. The resolver tries the recorded
selector, then the element's name, then its visible text, and reports which path
worked so the agent can tell a stale reference from a missing element.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeout

from cvnator.browser.controller import BrowserController
from cvnator.browser.perception import PageElement, PageState, observe
from cvnator.domain.actions import AgentDecision, ToolResult
from cvnator.observability import get_logger

log = get_logger(__name__)


class ToolExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class ResolvedTarget:
    element: PageElement | None
    selector: str
    strategy: str


class ToolContext:
    """Everything a tool may touch. Passed in, never reached for globally."""

    def __init__(
        self,
        browser: BrowserController,
        *,
        on_screenshot: Callable[[Path], Awaitable[None]] | None = None,
    ) -> None:
        self.browser = browser
        self.on_screenshot = on_screenshot
        self.state: PageState | None = None

    async def refresh(self) -> PageState:
        self.state = await observe(self.browser.page)
        return self.state


def resolve_target(state: PageState, decision: AgentDecision) -> ResolvedTarget:
    """Turn the agent's pointer into a selector, trying progressively looser matches."""
    target = decision.target

    if target.element_id:
        element = state.element_by_id(target.element_id)
        if element is not None:
            return ResolvedTarget(element, element.selector, "element_id")

    if target.selector:
        return ResolvedTarget(None, target.selector, "explicit_selector")

    if target.text:
        needle = target.text.strip().lower()
        for element in state.elements:
            haystack = f"{element.label} {element.text}".lower()
            if needle and needle in haystack:
                return ResolvedTarget(element, element.selector, "text_match")

    return ResolvedTarget(None, "", "unresolved")


async def _result(
    context: ToolContext,
    decision: AgentDecision,
    *,
    success: bool,
    url_before: str,
    fingerprint_before: str,
    observed: str = "",
    error: str = "",
    data: dict[str, Any] | None = None,
) -> ToolResult:
    """Re-observe after every action and report what genuinely changed."""
    state = await context.refresh()
    payload = dict(data or {})
    if state.alerts:
        payload["alerts"] = state.alerts
    return ToolResult(
        success=success,
        action=decision.action,
        error=error,
        observed_state=observed or state.title,
        url_before=url_before,
        url_after=state.url,
        dom_changed=state.fingerprint != fingerprint_before,
        page_fingerprint=state.fingerprint,
        data=payload,
    )


class BrowserToolbox:
    """Dispatches an :class:`AgentDecision` to a concrete browser action."""

    def __init__(self, context: ToolContext) -> None:
        self.context = context

    async def execute(self, decision: AgentDecision) -> ToolResult:
        state = self.context.state or await self.context.refresh()
        url_before = state.url
        fingerprint_before = state.fingerprint

        handler = getattr(self, f"_do_{decision.action.value}", None)
        if handler is None:
            return ToolResult(
                success=False,
                action=decision.action,
                error=f"No browser tool implements {decision.action.value}",
                url_before=url_before,
                url_after=url_before,
                page_fingerprint=fingerprint_before,
            )

        log.tool_call("%s %s", decision.action.value, decision.target.element_id or "")
        try:
            return await handler(decision, state, url_before, fingerprint_before)
        except PlaywrightTimeout as exc:
            return await _result(
                self.context, decision, success=False, url_before=url_before,
                fingerprint_before=fingerprint_before,
                error=f"Timed out: {exc}".split("\n")[0],
            )
        except (PlaywrightError, ToolExecutionError, FileNotFoundError, IndexError) as exc:
            return await _result(
                self.context, decision, success=False, url_before=url_before,
                fingerprint_before=fingerprint_before,
                error=str(exc).split("\n")[0],
            )

    # ----------------------------------------------------------------- perception

    async def _do_read_page(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        fresh = await self.context.refresh()
        return ToolResult(
            success=True,
            action=decision.action,
            observed_state=fresh.as_prompt_block(),
            url_before=url,
            url_after=fresh.url,
            dom_changed=fresh.fingerprint != fp,
            page_fingerprint=fresh.fingerprint,
            data={"element_count": len(fresh.elements), "alerts": fresh.alerts},
        )

    async def _do_read_accessibility_tree(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        from cvnator.browser.perception import accessibility_snapshot, flatten_accessibility

        snapshot = await accessibility_snapshot(self.context.browser.page)
        lines = flatten_accessibility(snapshot)
        return await _result(
            self.context, decision, success=bool(lines), url_before=url,
            fingerprint_before=fp,
            observed="\n".join(lines[:150]) or "Accessibility tree was empty",
            data={"node_count": len(lines)},
        )

    async def _do_get_url(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        current = await self.context.browser.current_url()
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=current, data={"tabs": self.context.browser.tab_urls()},
        )

    async def _do_scroll_page(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        direction = str(decision.args.get("direction", "down"))
        amount = int(decision.args.get("amount", 600))
        await self.context.browser.scroll(direction, amount)
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=f"Scrolled {direction} by {amount}px",
        )

    async def _do_wait(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        import asyncio

        seconds = min(float(decision.args.get("seconds", 1.5)), 10.0)
        await asyncio.sleep(seconds)
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=f"Waited {seconds}s",
        )

    async def _do_take_screenshot(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        path = await self.context.browser.screenshot(
            full_page=bool(decision.args.get("full_page", False))
        )
        if self.context.on_screenshot is not None:
            await self.context.on_screenshot(path)
        result = await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=f"Screenshot saved to {path.name}",
        )
        result.screenshot_reference = str(path)
        return result

    # ---------------------------------------------------------------- navigation

    async def _do_navigate(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        destination = decision.value or str(decision.args.get("url", ""))
        if not destination:
            raise ToolExecutionError("navigate requires a URL in `value` or `args.url`")
        nav = await self.context.browser.navigate(destination)
        return await _result(
            self.context, decision, success=nav.ok, url_before=url, fingerprint_before=fp,
            observed=f"HTTP {nav.status}" if nav.status else "navigated",
            error=nav.error, data={"status": nav.status},
        )

    async def _do_go_back(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        await self.context.browser.go_back()
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp
        )

    async def _do_open_tab(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        opened = await self.context.browser.open_tab(decision.value or "about:blank")
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=opened,
        )

    async def _do_close_tab(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        await self.context.browser.close_tab(decision.args.get("index"))
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp
        )

    async def _do_switch_tab(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        index = int(decision.args.get("index", 0))
        switched = await self.context.browser.switch_tab(index)
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=switched,
        )

    # --------------------------------------------------------------- interaction

    async def _do_click_element(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        resolved = resolve_target(state, decision)
        if not resolved.selector:
            return await _result(
                self.context, decision, success=False, url_before=url, fingerprint_before=fp,
                error=f"Could not resolve target {decision.target!r} on this page",
            )
        await self.context.browser.click(resolved.selector)

        # A click that changed nothing is not a success, whatever Playwright says.
        result = await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=f"Clicked via {resolved.strategy}",
            data={"selector": resolved.selector, "strategy": resolved.strategy},
        )
        if not result.dom_changed and not result.changed_url:
            result.success = False
            # The most common cause by far is HTML5 constraint validation refusing to
            # submit: the browser shows a native bubble that never reaches the DOM, so
            # name the offending fields explicitly or the agent just retries blindly.
            fresh = self.context.state
            invalid = [
                e for e in (fresh.elements if fresh else []) if e.invalid and e.validation_message
            ]
            if invalid:
                details = "; ".join(
                    f"[{e.element_id}] {e.label or e.name or 'field'}: {e.validation_message}"
                    for e in invalid[:5]
                )
                result.error = (
                    "The form did not submit because the browser rejected these fields: "
                    f"{details}. Correct the named values - do not click again."
                )
                result.data["invalid_fields"] = [e.element_id for e in invalid]
            else:
                result.error = (
                    "Click executed but the page did not change (same URL, same "
                    "fingerprint). The element may be inert, or a validation error may "
                    "be blocking progress."
                )
        return result

    async def _do_fill_field(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        resolved = resolve_target(state, decision)
        value = decision.value or ""
        if not resolved.selector:
            return await _result(
                self.context, decision, success=False, url_before=url, fingerprint_before=fp,
                error=f"Could not resolve field {decision.target!r}",
            )
        if resolved.element is not None and resolved.element.type == "file":
            return await _result(
                self.context, decision, success=False, url_before=url, fingerprint_before=fp,
                error=(
                    "That is a file input and cannot be filled with text. Use the "
                    "upload_file action on this element instead."
                ),
            )
        if resolved.element is not None and resolved.element.type == "password":
            return await _result(
                self.context, decision, success=False, url_before=url, fingerprint_before=fp,
                error=(
                    "Refusing to type into a password field from agent-supplied text. "
                    "Use the login action, which resolves credentials from the vault."
                ),
            )
        await self.context.browser.fill(resolved.selector, value)

        # Verify the field actually holds what we intended.
        fresh = await self.context.refresh()
        actual = None
        if resolved.element is not None:
            updated = next(
                (e for e in fresh.elements if e.selector == resolved.element.selector), None
            )
            actual = updated.value if updated else None
        matched = actual is None or actual.strip() == value.strip()
        return ToolResult(
            success=matched,
            action=decision.action,
            error="" if matched else f"Field holds {actual!r} after filling {value!r}",
            observed_state=f"Filled via {resolved.strategy}",
            url_before=url,
            url_after=fresh.url,
            dom_changed=fresh.fingerprint != fp,
            page_fingerprint=fresh.fingerprint,
            data={"selector": resolved.selector, "value_after": actual, "alerts": fresh.alerts},
        )

    async def _do_type_text(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        resolved = resolve_target(state, decision)
        if not resolved.selector:
            return await _result(
                self.context, decision, success=False, url_before=url, fingerprint_before=fp,
                error=f"Could not resolve field {decision.target!r}",
            )
        await self.context.browser.type_text(resolved.selector, decision.value or "")
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=f"Typed into {resolved.selector}",
        )

    async def _do_select_option(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        resolved = resolve_target(state, decision)
        if not resolved.selector:
            return await _result(
                self.context, decision, success=False, url_before=url, fingerprint_before=fp,
                error=f"Could not resolve select {decision.target!r}",
            )
        chosen = await self.context.browser.select_option(resolved.selector, decision.value or "")
        return await _result(
            self.context, decision, success=bool(chosen), url_before=url, fingerprint_before=fp,
            observed=f"Selected {chosen}",
            error="" if chosen else f"No option matched {decision.value!r}",
            data={"selected": chosen},
        )

    async def _do_set_checkbox(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        resolved = resolve_target(state, decision)
        if not resolved.selector:
            return await _result(
                self.context, decision, success=False, url_before=url, fingerprint_before=fp,
                error=f"Could not resolve checkbox {decision.target!r}",
            )
        wanted = decision.args.get("checked", decision.value not in {"false", "no", "0"})
        await self.context.browser.set_checkbox(resolved.selector, bool(wanted))
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp,
            observed=f"Checkbox set to {bool(wanted)}",
        )

    async def _do_press_key(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        await self.context.browser.press_key(decision.value or "Enter")
        return await _result(
            self.context, decision, success=True, url_before=url, fingerprint_before=fp
        )

    async def _do_upload_file(
        self, decision: AgentDecision, state: PageState, url: str, fp: str
    ) -> ToolResult:
        resolved = resolve_target(state, decision)
        raw_path = decision.value or str(decision.args.get("path", ""))
        if not raw_path:
            raise ToolExecutionError("upload_file requires a file path")
        if not resolved.selector:
            file_inputs = [e for e in state.elements if e.type == "file"]
            if not file_inputs:
                return await _result(
                    self.context, decision, success=False, url_before=url, fingerprint_before=fp,
                    error="No file input found on the page",
                )
            resolved = ResolvedTarget(file_inputs[0], file_inputs[0].selector, "only_file_input")

        await self.context.browser.upload_file(resolved.selector, Path(raw_path))

        # Confirm the page acknowledged the file rather than trusting the call.
        fresh = await self.context.refresh()
        filename = Path(raw_path).name
        acknowledged = filename.lower() in fresh.text.lower() or any(
            filename.lower() in (e.value or "").lower() for e in fresh.elements
        )
        return ToolResult(
            success=True,
            action=decision.action,
            observed_state=(
                f"Uploaded {filename}"
                + ("; page shows the filename" if acknowledged else "; page does not echo the filename yet")
            ),
            url_before=url,
            url_after=fresh.url,
            dom_changed=fresh.fingerprint != fp,
            page_fingerprint=fresh.fingerprint,
            data={"file": filename, "acknowledged": acknowledged, "alerts": fresh.alerts},
        )

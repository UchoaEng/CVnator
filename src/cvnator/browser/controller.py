"""Playwright/Chromium controller.

Responsibilities kept here (and nowhere else):

* browser lifecycle, contexts and tabs
* a persistent profile directory, so a login the *user* performs by hand survives
  across sessions and the agent does not have to re-authenticate constantly
* typing secrets without ever exposing them to the agent's context
* screenshots, registered as disposable artifacts

Default is headful. During development a visible browser is worth far more than the
resources it costs: the user can watch, take over, solve a CAPTCHA, and see exactly
what the agent saw when something goes wrong.

No stealth plugins, no fingerprint spoofing, no anti-bot evasion. If a site blocks
automation, the correct behaviour is to stop and tell the user, not to hide.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeout,
)

from cvnator.config import Settings, get_settings
from cvnator.observability import get_logger

log = get_logger(__name__)


class BrowserNotStarted(RuntimeError):
    pass


@dataclass(slots=True)
class NavigationResult:
    url: str
    status: int | None
    ok: bool
    error: str = ""


class BrowserController:
    def __init__(self, settings: Settings | None = None, *, headless: bool | None = None) -> None:
        self.settings = settings or get_settings()
        self.headless = self.settings.browser_headless if headless is None else headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._user_has_control = False
        self._screenshot_counter = 0

    # ------------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._context is not None:
            return
        self._playwright = await async_playwright().start()
        profile_dir = self.settings.browser_profile_dir
        profile_dir.mkdir(parents=True, exist_ok=True)

        # A persistent context keeps cookies and storage between runs, so the user's
        # own manual logins are reused instead of being re-prompted every session.
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=self.headless,
            slow_mo=self.settings.browser_slowmo_ms or 0,
            viewport={"width": 1440, "height": 900},
            accept_downloads=True,
            # No stealth or anti-detection flags, deliberately. If a site declines to
            # be automated, the agent escalates to the user instead of hiding.
        )
        self._context.set_default_timeout(self.settings.browser_action_timeout_ms)
        self._context.set_default_navigation_timeout(self.settings.browser_nav_timeout_ms)
        pages = self._context.pages
        self._page = pages[0] if pages else await self._context.new_page()
        log.info("Browser started (headless=%s, profile=%s)", self.headless, profile_dir)

    async def stop(self) -> None:
        try:
            if self._context is not None:
                await self._context.close()
        finally:
            self._context = None
            self._page = None
            if self._browser is not None:
                await self._browser.close()
                self._browser = None
            if self._playwright is not None:
                await self._playwright.stop()
                self._playwright = None
            log.info("Browser stopped")

    @property
    def started(self) -> bool:
        return self._context is not None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise BrowserNotStarted("Browser has not been started")
        return self._page

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise BrowserNotStarted("Browser has not been started")
        return self._context

    # ----------------------------------------------------------------- user takeover

    @property
    def user_has_control(self) -> bool:
        return self._user_has_control

    def grant_user_control(self) -> None:
        """Agent stops acting; the browser stays open for the user to drive."""
        self._user_has_control = True
        log.needs_user("User has taken control of the browser; agent actions are suspended")

    def reclaim_control(self) -> None:
        """Agent resumes. It must re-observe before acting; state may have changed."""
        self._user_has_control = False
        log.info("Agent has reclaimed control of the browser")

    def _assert_agent_control(self) -> None:
        if self._user_has_control:
            raise PermissionError(
                "The user currently has control of the browser. "
                "Call reclaim_control() before acting."
            )

    # -------------------------------------------------------------------- navigation

    async def navigate(self, url: str, *, wait_until: str = "domcontentloaded") -> NavigationResult:
        self._assert_agent_control()
        try:
            response = await self.page.goto(url, wait_until=wait_until)  # type: ignore[arg-type]
            await self._settle()
            return NavigationResult(
                url=self.page.url,
                status=response.status if response else None,
                ok=bool(response is None or response.ok),
            )
        except PlaywrightTimeout as exc:
            return NavigationResult(url=self.page.url, status=None, ok=False, error=str(exc))

    async def go_back(self) -> NavigationResult:
        self._assert_agent_control()
        await self.page.go_back()
        await self._settle()
        return NavigationResult(url=self.page.url, status=None, ok=True)

    async def _settle(self, timeout_ms: int = 4000) -> None:
        """Best-effort wait for the network to go quiet. Never fatal on timeout."""
        # A page that never goes idle (polling, websockets) is normal, not an error.
        with contextlib.suppress(PlaywrightTimeout):
            await self.page.wait_for_load_state("networkidle", timeout=timeout_ms)

    # -------------------------------------------------------------------------- tabs

    async def open_tab(self, url: str = "about:blank") -> str:
        self._assert_agent_control()
        page = await self.context.new_page()
        if url and url != "about:blank":
            await page.goto(url)
        self._page = page
        return page.url

    async def close_tab(self, index: int | None = None) -> None:
        self._assert_agent_control()
        pages = self.context.pages
        target = self._page if index is None else pages[index]
        if target is not None and len(pages) > 1:
            await target.close()
            self._page = self.context.pages[-1]

    async def switch_tab(self, index: int) -> str:
        self._assert_agent_control()
        pages = self.context.pages
        if not 0 <= index < len(pages):
            raise IndexError(f"No tab at index {index} (open tabs: {len(pages)})")
        self._page = pages[index]
        await self._page.bring_to_front()
        return self._page.url

    def tab_urls(self) -> list[str]:
        return [p.url for p in self.context.pages] if self._context else []

    async def adopt_newest_tab(self) -> bool:
        """Follow a link that opened a new tab (common on external "Apply" buttons)."""
        pages = self.context.pages
        if pages and pages[-1] is not self._page:
            self._page = pages[-1]
            await self._page.bring_to_front()
            await self._settle()
            log.info("Followed application to a new tab: %s", self._page.url)
            return True
        return False

    # ----------------------------------------------------------------------- actions

    def _locator(self, selector: str):
        return self.page.locator(selector).first

    async def click(self, selector: str, *, timeout_ms: int | None = None) -> None:
        self._assert_agent_control()
        await self._locator(selector).click(
            timeout=timeout_ms or self.settings.browser_action_timeout_ms
        )
        await self._settle(2500)

    async def fill(self, selector: str, value: str) -> None:
        self._assert_agent_control()
        locator = self._locator(selector)
        await locator.fill(value)

    async def type_text(self, selector: str, value: str, delay_ms: int = 25) -> None:
        self._assert_agent_control()
        locator = self._locator(selector)
        await locator.click()
        await locator.press_sequentially(value, delay=delay_ms)

    async def fill_secret(self, selector: str, secret: str) -> None:
        """Type a credential.

        ``secret`` reaches this method straight from the vault. It is never returned,
        logged, or placed in any structure the agent can read.
        """
        self._assert_agent_control()
        await self._locator(selector).fill(secret)
        log.info("Filled a credential field (value not logged)")

    async def select_option(self, selector: str, value: str) -> list[str]:
        self._assert_agent_control()
        locator = self._locator(selector)
        try:
            return await locator.select_option(value=value)
        except Exception:
            # Many forms label options differently from their values.
            return await locator.select_option(label=value)

    async def set_checkbox(self, selector: str, checked: bool) -> None:
        self._assert_agent_control()
        locator = self._locator(selector)
        if checked:
            await locator.check()
        else:
            await locator.uncheck()

    async def press_key(self, key: str) -> None:
        self._assert_agent_control()
        await self.page.keyboard.press(key)
        await self._settle(1500)

    async def scroll(self, direction: str = "down", amount: int = 600) -> None:
        self._assert_agent_control()
        delta = amount if direction == "down" else -amount
        await self.page.mouse.wheel(0, delta)
        await asyncio.sleep(0.25)

    async def upload_file(self, selector: str, file_path: Path) -> None:
        self._assert_agent_control()
        resolved = await asyncio.to_thread(Path(file_path).resolve)
        if not await asyncio.to_thread(resolved.exists):
            raise FileNotFoundError(f"Cannot upload missing file: {resolved}")
        await self._locator(selector).set_input_files(str(resolved))
        await self._settle(2500)

    async def screenshot(self, *, full_page: bool = False) -> Path:
        self._screenshot_counter += 1
        directory = self.settings.screenshots_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"shot_{self._screenshot_counter:05d}.png"
        await self.page.screenshot(path=str(path), full_page=full_page)
        return path

    async def current_url(self) -> str:
        return self.page.url

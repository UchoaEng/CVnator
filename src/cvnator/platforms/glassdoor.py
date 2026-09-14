"""Glassdoor adapter.

**Status: scaffold, not validated against the live site.**

Glassdoor gates most job content behind an account and is aggressive about
automation. The same rules apply as everywhere else: no evasion, and a block or a
login wall becomes an escalation to the user rather than something to work around.
Ships unregistered.
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus, urlsplit

from cvnator.browser.perception import observe
from cvnator.domain.job import ApplyMethod, JobPosting
from cvnator.domain.search import SearchProfile
from cvnator.observability import get_logger
from cvnator.platforms.base import PlatformAdapter

log = get_logger(__name__)

_WALL_MARKERS = ("sign in to continue", "create your account", "entre para continuar")


class GlassdoorAdapter(PlatformAdapter):
    name = "glassdoor"
    domains = ("glassdoor.com", "glassdoor.com.br")
    requires_login = True

    def __init__(self, browser, host: str = "https://www.glassdoor.com.br") -> None:
        super().__init__(browser)
        self.host = host.rstrip("/")

    async def is_logged_in(self) -> bool:
        state = await observe(self.browser.page)
        lowered = state.text[:3000].lower()
        return not any(marker in lowered for marker in _WALL_MARKERS)

    async def discover(self, search: SearchProfile, *, limit: int = 20) -> list[JobPosting]:
        query = quote_plus(search.role_title or search.name)
        result = await self.browser.navigate(f"{self.host}/Vaga/index.htm?keyword={query}")
        if not result.ok:
            log.warning("Glassdoor search did not load: %s", result.error)
            return []

        if not await self.is_logged_in():
            log.needs_user(
                "Glassdoor is showing a sign-in wall. Sign in yourself in the open browser; "
                "the session is stored in the persistent profile and reused."
            )
            return []

        state = await observe(self.browser.page, max_elements=200)
        jobs: list[JobPosting] = []
        seen: set[str] = set()
        for element in state.elements:
            if element.tag != "a" or not element.href:
                continue
            match = re.search(r"jobListingId=(\d+)", element.href)
            if not match or match.group(1) in seen:
                continue
            seen.add(match.group(1))
            href = element.href
            jobs.append(
                JobPosting(
                    platform=self.name,
                    platform_job_id=match.group(1),
                    url=href if href.startswith("http") else f"{self.host}{href}",
                    role=element.text.strip()[:200],
                )
            )
            if len(jobs) >= limit:
                break
        log.info("Glassdoor discovery found %s postings", len(jobs))
        return jobs

    async def open_job(self, job: JobPosting) -> bool:
        return (await self.browser.navigate(job.url)).ok

    async def enrich(self, job: JobPosting) -> JobPosting:
        state = await observe(self.browser.page)
        job.description = state.text[:20000]
        job.application_platform = self.name
        job.language = job.language or state.lang[:2]
        return job

    async def start_application(self, job: JobPosting) -> ApplyMethod:
        state = await observe(self.browser.page)
        button = next(
            (
                e
                for e in state.elements
                if e.tag in {"button", "a"}
                and re.search(r"apply|candidat", f"{e.text} {e.label}", re.I)
            ),
            None,
        )
        if button is None:
            return ApplyMethod.UNKNOWN
        before = urlsplit(state.url).netloc
        await self.browser.click(button.selector)
        if await self.browser.adopt_newest_tab():
            return ApplyMethod.EXTERNAL
        after = await observe(self.browser.page)
        return (
            ApplyMethod.EXTERNAL
            if urlsplit(after.url).netloc != before
            else ApplyMethod.NATIVE_FORM
        )


# Not registered on import; see the note in linkedin.py.

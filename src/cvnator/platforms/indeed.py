"""Indeed adapter.

**Status: scaffold, not validated against the live site.**

The same terms-of-service caveat as LinkedIn applies: Indeed's terms prohibit
automated access, and this adapter ships unregistered for that reason. There is no
detection-avoidance code here and there will not be; a block results in an escalation
to the user, not a workaround.

Indeed routes many applications to the employer's own ATS, so ``start_application``
follows that handoff rather than treating a departure from indeed.com as a failure.
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


class IndeedAdapter(PlatformAdapter):
    name = "indeed"
    domains = ("indeed.com", "br.indeed.com")
    requires_login = False

    def __init__(self, browser, host: str = "https://br.indeed.com") -> None:
        super().__init__(browser)
        self.host = host.rstrip("/")

    def build_search_url(self, search: SearchProfile) -> str:
        params = [f"q={quote_plus(search.role_title or search.name)}"]
        if search.location:
            params.append(f"l={quote_plus(search.location)}")
        joined = "&".join(params)
        return f"{self.host}/jobs?{joined}"

    async def discover(self, search: SearchProfile, *, limit: int = 20) -> list[JobPosting]:
        result = await self.browser.navigate(self.build_search_url(search))
        if not result.ok:
            log.warning("Indeed search did not load: %s", result.error)
            return []

        state = await observe(self.browser.page, max_elements=200)
        jobs: list[JobPosting] = []
        seen: set[str] = set()
        for element in state.elements:
            if element.tag != "a" or not element.href:
                continue
            match = re.search(r"[?&]jk=([0-9a-f]+)", element.href)
            if not match or match.group(1) in seen:
                continue
            seen.add(match.group(1))
            jobs.append(
                JobPosting(
                    platform=self.name,
                    platform_job_id=match.group(1),
                    url=f"{self.host}/viewjob?jk={match.group(1)}",
                    role=element.text.strip()[:200],
                )
            )
            if len(jobs) >= limit:
                break
        log.info("Indeed discovery found %s postings", len(jobs))
        return jobs

    async def open_job(self, job: JobPosting) -> bool:
        return (await self.browser.navigate(job.url)).ok

    async def enrich(self, job: JobPosting) -> JobPosting:
        state = await observe(self.browser.page)
        job.description = state.text[:20000]
        job.application_platform = self.name
        job.language = job.language or state.lang[:2]
        if not job.role:
            job.role = next((e.text.strip() for e in state.elements if e.tag == "h1"), state.title)
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

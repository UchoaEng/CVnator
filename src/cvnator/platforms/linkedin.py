"""LinkedIn adapter.

**Status: scaffold, not validated against the live site.** Everything here is
written from the public structure of LinkedIn's job pages, but it has never been run
against production, and LinkedIn changes its markup often. Treat the selectors as
hints; the agent's generic path is what actually carries an application through.

Read this before enabling it:

* Automating LinkedIn is against LinkedIn's User Agreement. Running this against a
  real account risks that account being restricted. That is the user's decision to
  make knowingly, which is why nothing here is enabled by default and why the
  adapter is not registered unless explicitly imported.
* There is no anti-detection code here and there will not be. If LinkedIn presents a
  challenge or a block, the agent stops and asks the user - see
  ``cvnator.agent.pacing.detect_block``.
* Authentication is the user's job. The browser uses a persistent profile, so a
  login the user performs by hand is reused; the agent never types a password it was
  not explicitly given through the vault.
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus, urlsplit

from cvnator.browser.perception import observe
from cvnator.domain.job import ApplyMethod, JobPosting
from cvnator.domain.profile import RemotePolicy
from cvnator.domain.search import SearchProfile
from cvnator.observability import get_logger
from cvnator.platforms.base import PlatformAdapter

log = get_logger(__name__)

SEARCH_URL = "https://www.linkedin.com/jobs/search/"

#: LinkedIn's `f_WT` filter values.
_REMOTE_FILTER = {RemotePolicy.ONSITE: "1", RemotePolicy.REMOTE: "2", RemotePolicy.HYBRID: "3"}

_SIGNED_OUT_MARKERS = (
    "sign in", "join now", "entrar", "cadastre-se", "iniciar sesión",
)


class LinkedInAdapter(PlatformAdapter):
    name = "linkedin"
    domains = ("linkedin.com",)
    requires_login = True

    def build_search_url(self, search: SearchProfile, start: int = 0) -> str:
        params = [f"keywords={quote_plus(search.role_title or search.name)}"]
        if search.location:
            params.append(f"location={quote_plus(search.location)}")
        remote = [
            _REMOTE_FILTER[p] for p in search.remote_policies if p in _REMOTE_FILTER
        ]
        if remote:
            params.append(f"f_WT={'%2C'.join(remote)}")
        if start:
            params.append(f"start={start}")
        return f"{SEARCH_URL}?{'&'.join(params)}"

    async def is_logged_in(self) -> bool:
        state = await observe(self.browser.page)
        lowered = state.text[:2000].lower()
        return not any(marker in lowered for marker in _SIGNED_OUT_MARKERS)

    async def discover(self, search: SearchProfile, *, limit: int = 20) -> list[JobPosting]:
        result = await self.browser.navigate(self.build_search_url(search))
        if not result.ok:
            log.warning("LinkedIn search did not load: %s", result.error)
            return []

        if not await self.is_logged_in():
            log.needs_user(
                "LinkedIn appears to be signed out. Sign in yourself in the open browser; "
                "the session is stored in the persistent profile and reused."
            )
            return []

        state = await observe(self.browser.page, max_elements=200)
        jobs: list[JobPosting] = []
        seen: set[str] = set()
        for element in state.elements:
            if element.tag != "a" or not element.href:
                continue
            match = re.search(r"/jobs/view/(\d+)", element.href)
            if not match or match.group(1) in seen:
                continue
            seen.add(match.group(1))
            jobs.append(
                JobPosting(
                    platform=self.name,
                    platform_job_id=match.group(1),
                    url=f"https://www.linkedin.com/jobs/view/{match.group(1)}/",
                    role=element.text.strip()[:200],
                )
            )
            if len(jobs) >= limit:
                break

        log.info("LinkedIn discovery found %s postings", len(jobs))
        return jobs

    async def open_job(self, job: JobPosting) -> bool:
        return (await self.browser.navigate(job.url)).ok

    async def enrich(self, job: JobPosting) -> JobPosting:
        state = await observe(self.browser.page)
        job.description = state.text[:20000]
        job.application_platform = self.name
        job.language = job.language or state.lang[:2]

        lowered = state.text.lower()
        if "remote" in lowered or "remoto" in lowered:
            job.remote_policy = RemotePolicy.REMOTE
        elif "hybrid" in lowered or "híbrido" in lowered:
            job.remote_policy = RemotePolicy.HYBRID

        # Easy Apply keeps the application on LinkedIn; anything else leaves it.
        easy = any(
            "easy apply" in f"{e.text} {e.label}".lower()
            or "candidatura simplificada" in f"{e.text} {e.label}".lower()
            for e in state.elements
        )
        job.apply_method = ApplyMethod.EASY_APPLY if easy else ApplyMethod.EXTERNAL
        return job

    async def start_application(self, job: JobPosting) -> ApplyMethod:
        """Press the apply control, and follow the application if it leaves LinkedIn.

        Leaving LinkedIn is normal, not a failure: most postings hand off to the
        employer's own ATS. The orchestrator re-identifies the destination and the
        generic agent continues there.
        """
        state = await observe(self.browser.page)
        button = next(
            (
                e
                for e in state.elements
                if e.tag in {"button", "a"}
                and re.search(r"\b(easy apply|apply|candidatar)\b", f"{e.text} {e.label}", re.I)
            ),
            None,
        )
        if button is None:
            log.info("No LinkedIn apply control recognised; deferring to the generic agent")
            return ApplyMethod.UNKNOWN

        before_host = urlsplit(state.url).netloc
        await self.browser.click(button.selector)

        if await self.browser.adopt_newest_tab():
            after = await self.browser.current_url()
            from cvnator.platforms.generic import identify_ats

            job.application_platform = identify_ats(after)
            log.info("LinkedIn handed off to %s (%s)", job.application_platform, after)
            return ApplyMethod.EXTERNAL

        after_state = await observe(self.browser.page)
        if urlsplit(after_state.url).netloc != before_host:
            from cvnator.platforms.generic import identify_ats

            job.application_platform = identify_ats(after_state.url)
            return ApplyMethod.EXTERNAL
        return ApplyMethod.EASY_APPLY


# Deliberately NOT registered on import. Enabling LinkedIn is an explicit act:
#     from cvnator.platforms.base import registry
#     from cvnator.platforms.linkedin import LinkedInAdapter
#     registry.register(LinkedInAdapter)

"""Adapter for the local MockATS.

This is a real, fully working adapter - it just points at the test harness instead
of a third-party site. It exists so the whole pipeline (discovery, parsing, matching,
resume generation, form completion, submission validation, history) can be exercised
end to end, repeatedly and deterministically, without depending on anyone's
production service.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

from cvnator.browser.perception import observe
from cvnator.domain.job import ApplyMethod, JobPosting
from cvnator.domain.profile import RemotePolicy
from cvnator.domain.search import SearchProfile
from cvnator.observability import get_logger
from cvnator.platforms.base import PlatformAdapter, registry

log = get_logger(__name__)

MOCK_BASE_URL = "http://127.0.0.1:8099"

_REMOTE_MAP = {
    "remote": RemotePolicy.REMOTE,
    "hybrid": RemotePolicy.HYBRID,
    "onsite": RemotePolicy.ONSITE,
}


class MockATSAdapter(PlatformAdapter):
    name = "mock"
    domains = ("127.0.0.1:8099", "localhost:8099", "127.0.0.1", "localhost")
    requires_login = False

    def __init__(self, browser, base_url: str = MOCK_BASE_URL) -> None:
        super().__init__(browser)
        self.base_url = base_url.rstrip("/")

    @classmethod
    def handles(cls, url: str) -> bool:
        return "127.0.0.1:8099" in url or "localhost:8099" in url

    async def discover(self, search: SearchProfile, *, limit: int = 20) -> list[JobPosting]:
        result = await self.browser.navigate(f"{self.base_url}/jobs")
        if not result.ok:
            log.warning("MockATS listing unreachable at %s: %s", self.base_url, result.error)
            return []

        state = await observe(self.browser.page)
        jobs: list[JobPosting] = []
        for element in state.elements:
            if element.tag != "a" or not element.href:
                continue
            match = re.search(r"/jobs/([A-Za-z0-9\-]+)$", element.href)
            if not match:
                continue
            url = urljoin(self.base_url + "/", element.href.lstrip("/"))
            jobs.append(
                JobPosting(
                    platform=self.name,
                    platform_job_id=match.group(1),
                    url=url,
                    apply_url=url,
                    role=element.text.strip(),
                    apply_method=ApplyMethod.NATIVE_FORM,
                )
            )
            if len(jobs) >= limit:
                break

        log.info("MockATS discovery found %s postings", len(jobs))
        return jobs

    async def open_job(self, job: JobPosting) -> bool:
        result = await self.browser.navigate(job.url)
        return result.ok

    async def enrich(self, job: JobPosting) -> JobPosting:
        """Read the posting page for the fields the listing page does not carry."""
        state = await observe(self.browser.page)
        text = state.text

        title = next(
            (e.text.strip() for e in state.elements if e.tag in {"h1"} and e.text.strip()), ""
        )
        if title:
            job.role = title

        # The mock renders a single meta line: "Company · Location · remote · ref X · posted Y"
        meta = re.search(
            r"^(?P<company>[^·\n]+)·(?P<location>[^·\n]+)·(?P<remote>[^·\n]+)"
            r"·\s*ref\s*(?P<ref>[^·\n]+)·\s*posted\s*(?P<posted>[^\n]+)",
            text,
            re.MULTILINE,
        )
        if meta:
            job.company = meta.group("company").strip()
            job.location = meta.group("location").strip()
            job.remote_policy = _REMOTE_MAP.get(
                meta.group("remote").strip().lower(), RemotePolicy.ANY
            )
            job.platform_job_id = meta.group("ref").strip() or job.platform_job_id
            job.posted_at = meta.group("posted").strip()

        # Everything after the meta line down to the apply control is the description.
        description = text
        marker = text.find("Apply for this position")
        if marker > 0:
            description = text[:marker]
        job.description = description.strip()
        job.application_platform = self.name
        job.apply_method = ApplyMethod.NATIVE_FORM
        return job

    async def start_application(self, job: JobPosting) -> ApplyMethod:
        state = await observe(self.browser.page)
        apply_button = next(
            (
                e
                for e in state.elements
                if e.tag == "button" and "apply" in f"{e.text} {e.label}".lower()
            ),
            None,
        )
        if apply_button is None:
            log.warning("MockATS apply control not found; handing over to the generic agent")
            return ApplyMethod.UNKNOWN
        await self.browser.click(apply_button.selector)
        return ApplyMethod.NATIVE_FORM


registry.register(MockATSAdapter)

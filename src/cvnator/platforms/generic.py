"""Generic ATS adapter - the fallback that makes an unknown platform survivable.

When a posting sends the agent somewhere nobody wrote an adapter for, this takes
over. It knows nothing platform-specific. It reads the page, works out whether an
application is possible, finds whatever control starts one, and hands off to the
agent loop, which drives the rest from the DOM.

It cannot *discover* jobs - searching needs a search interface, and there is no
generic one - so ``discover`` returns an empty list rather than pretending.
"""

from __future__ import annotations

import re

from cvnator.browser.perception import observe
from cvnator.domain.job import ApplyMethod, JobPosting
from cvnator.domain.search import SearchProfile
from cvnator.observability import get_logger
from cvnator.platforms.base import PlatformAdapter, registry

log = get_logger(__name__)

#: Words that start an application, across the languages the product targets.
_APPLY_WORDS = (
    "apply", "apply now", "easy apply", "submit application", "start application",
    "candidatar", "candidate-se", "candidatar-se", "quero me candidatar", "inscrever",
    "postular", "aplicar", "enviar candidatura", "participar do processo",
)

#: Hosts of well-known ATS products, used only to *label* the platform in history.
KNOWN_ATS_HOSTS: dict[str, str] = {
    "greenhouse.io": "greenhouse",
    "boards.greenhouse.io": "greenhouse",
    "lever.co": "lever",
    "jobs.lever.co": "lever",
    "myworkdayjobs.com": "workday",
    "wd1.myworkdayjobs.com": "workday",
    "ashbyhq.com": "ashby",
    "jobs.ashbyhq.com": "ashby",
    "smartrecruiters.com": "smartrecruiters",
    "taleo.net": "taleo",
    "successfactors.com": "successfactors",
    "successfactors.eu": "successfactors",
    "icims.com": "icims",
    "workable.com": "workable",
    "breezy.hr": "breezy",
    "recruitee.com": "recruitee",
    "bamboohr.com": "bamboohr",
    "gupy.io": "gupy",
    "abler.com.br": "abler",
    "solides.com": "solides",
}


def identify_ats(url: str) -> str:
    """Best-effort label for the ATS behind a URL. Never used to gate behaviour."""
    from urllib.parse import urlsplit

    host = urlsplit(url).netloc.lower().removeprefix("www.")
    for domain, label in KNOWN_ATS_HOSTS.items():
        if host == domain or host.endswith(f".{domain}"):
            return label
    return "unknown"


class GenericATSAdapter(PlatformAdapter):
    name = "generic"
    domains = ()
    requires_login = False

    @classmethod
    def handles(cls, url: str) -> bool:
        # The generic adapter is chosen by the registry as a fallback, never matched.
        return False

    async def discover(self, search: SearchProfile, *, limit: int = 20) -> list[JobPosting]:
        log.info(
            "Generic adapter cannot search: there is no platform-independent search "
            "interface. Discovery must come from a specific adapter or a supplied URL."
        )
        return []

    async def open_job(self, job: JobPosting) -> bool:
        result = await self.browser.navigate(job.apply_url or job.url)
        return result.ok

    async def enrich(self, job: JobPosting) -> JobPosting:
        state = await observe(self.browser.page)
        job.description = job.description or state.text[:20000]
        job.application_platform = identify_ats(state.url)
        job.language = job.language or state.lang[:2]
        if not job.role:
            heading = next((e.text.strip() for e in state.elements if e.tag == "h1"), "")
            job.role = heading or state.title
        return job

    async def start_application(self, job: JobPosting) -> ApplyMethod:
        """Find and press whatever starts an application here."""
        state = await observe(self.browser.page)
        candidates = [
            element
            for element in state.elements
            if element.tag in {"button", "a"} or element.role == "button"
            if any(
                re.search(rf"\b{re.escape(word)}\b", f"{element.text} {element.label}".lower())
                for word in _APPLY_WORDS
            )
        ]
        if not candidates:
            log.info("No apply control recognised; the agent loop will reason about the page")
            return ApplyMethod.UNKNOWN

        # Prefer a button over a link: a link is more likely to be a listing entry.
        candidates.sort(key=lambda e: (e.tag != "button", len(e.text)))
        chosen = candidates[0]
        before = state.url
        await self.browser.click(chosen.selector)

        # An external apply usually opens a new tab or leaves the domain entirely.
        if await self.browser.adopt_newest_tab():
            after = await self.browser.current_url()
            log.info("Application continued on a new tab: %s", identify_ats(after))
            return ApplyMethod.EXTERNAL

        after_state = await observe(self.browser.page)
        from urllib.parse import urlsplit

        if urlsplit(after_state.url).netloc != urlsplit(before).netloc:
            log.info(
                "Application redirected off-platform to %s (%s)",
                urlsplit(after_state.url).netloc,
                identify_ats(after_state.url),
            )
            return ApplyMethod.EXTERNAL
        return ApplyMethod.NATIVE_FORM


registry.register(GenericATSAdapter, fallback=True)

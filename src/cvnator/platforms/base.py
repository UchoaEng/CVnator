"""Platform adapters.

An adapter is an *accelerator*, not a requirement. It encodes what we already know
about a platform - where its search lives, how its result list is shaped, which
button starts an application - so the agent does not have to rediscover it by
reasoning on every run. That saves time, tokens and mistakes.

What an adapter must never do is become a precondition. Every adapter method may
return "I don't know", and the generic, AI-driven path takes over from there. That
is what keeps an unknown ATS from being a dead end: the worst case is that the agent
works it out from the DOM, exactly as it would a known platform whose markup just
changed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from urllib.parse import urlsplit

from cvnator.browser.controller import BrowserController
from cvnator.domain.job import ApplyMethod, JobPosting
from cvnator.domain.search import SearchProfile
from cvnator.observability import get_logger

log = get_logger(__name__)


class PlatformAdapter(ABC):
    """Base for every platform integration."""

    #: Short stable identifier stored on jobs and applications.
    name: str = "generic"
    #: Hostnames this adapter recognises.
    domains: tuple[str, ...] = ()
    #: True when the platform generally requires an authenticated session.
    requires_login: bool = False

    def __init__(self, browser: BrowserController) -> None:
        self.browser = browser

    @classmethod
    def handles(cls, url: str) -> bool:
        host = urlsplit(url).netloc.lower().removeprefix("www.")
        return any(host == d or host.endswith(f".{d}") for d in cls.domains)

    @abstractmethod
    async def discover(self, search: SearchProfile, *, limit: int = 20) -> list[JobPosting]:
        """Find postings matching ``search``. May return an empty list."""

    @abstractmethod
    async def open_job(self, job: JobPosting) -> bool:
        """Navigate to the posting. Returns False if it could not be opened."""

    async def enrich(self, job: JobPosting) -> JobPosting:
        """Fill in details only available on the posting page. Optional."""
        return job

    async def start_application(self, job: JobPosting) -> ApplyMethod:
        """Begin the application, returning how it is being applied to.

        The default returns ``UNKNOWN``, which tells the orchestrator to let the
        generic agent find the apply control itself.
        """
        return ApplyMethod.UNKNOWN

    async def is_logged_in(self) -> bool:
        return not self.requires_login


class AdapterRegistry:
    """Resolves a URL or platform name to the most specific adapter available."""

    def __init__(self) -> None:
        self._adapters: list[type[PlatformAdapter]] = []
        self._fallback: type[PlatformAdapter] | None = None

    def register(self, adapter: type[PlatformAdapter], *, fallback: bool = False) -> None:
        if fallback:
            self._fallback = adapter
        else:
            self._adapters.append(adapter)

    def for_url(self, url: str, browser: BrowserController) -> PlatformAdapter:
        for adapter in self._adapters:
            if adapter.handles(url):
                return adapter(browser)
        if self._fallback is None:
            raise LookupError("No adapter registered and no generic fallback available")
        log.info("No specific adapter for %s; using the generic AI-driven path", url)
        return self._fallback(browser)

    def for_name(self, name: str, browser: BrowserController) -> PlatformAdapter:
        for adapter in self._adapters:
            if adapter.name == name:
                return adapter(browser)
        if self._fallback is None:
            raise LookupError(f"No adapter named {name!r}")
        return self._fallback(browser)

    def names(self) -> list[str]:
        names = [a.name for a in self._adapters]
        if self._fallback is not None:
            names.append(self._fallback.name)
        return names


registry = AdapterRegistry()

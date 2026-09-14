"""Rate limiting and pacing.

The goal is a system that behaves reasonably: steady, bounded, and not a burden on
anyone's servers. Concretely that means a per-platform minimum gap between actions,
a randomised cooldown between applications, a daily cap, and exponential backoff
after errors.

This is explicitly *not* evasion. The randomisation exists so the agent does not
hammer a server on a fixed metronome, not to imitate a human or defeat bot
detection. If a platform signals that it does not want this traffic, the agent's
answer is to stop and tell the user - see ``BlockDetected``.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from cvnator.observability import get_logger

log = get_logger(__name__)

_BLOCK_PATTERNS = (
    r"unusual (traffic|activity)",
    r"(temporarily|permanently) (blocked|restricted)",
    r"rate limit(ed)?",
    r"too many requests",
    r"automated (traffic|queries|access)",
    r"access denied",
    r"your account has been restricted",
    r"atividade (incomum|suspeita)",
    r"muitas (requisi[çc][õo]es|tentativas)",
    r"acesso (negado|bloqueado)",
)
_BLOCK_RE = re.compile("|".join(_BLOCK_PATTERNS), re.IGNORECASE)


class BlockDetected(RuntimeError):
    """The platform has told us to stop. We stop, and the user is informed."""


def detect_block(page_text: str) -> str:
    match = _BLOCK_RE.search(page_text or "")
    return match.group(0) if match else ""


@dataclass
class PlatformPace:
    min_action_gap_s: float = 1.5
    min_application_gap_s: float = 25.0
    max_application_gap_s: float = 75.0
    daily_limit: int = 30


@dataclass
class RateLimiter:
    """Per-platform pacing with error backoff and a daily cap."""

    default: PlatformPace = field(default_factory=PlatformPace)
    overrides: dict[str, PlatformPace] = field(default_factory=dict)

    _last_action: dict[str, float] = field(default_factory=dict)
    _last_application: dict[str, float] = field(default_factory=dict)
    _daily_counts: dict[tuple[str, date], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _consecutive_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def pace_for(self, platform: str) -> PlatformPace:
        return self.overrides.get(platform, self.default)

    async def before_action(self, platform: str) -> None:
        pace = self.pace_for(platform)
        last = self._last_action.get(platform)
        now = time.monotonic()
        if last is not None:
            wait = pace.min_action_gap_s - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_action[platform] = time.monotonic()

    async def before_application(self, platform: str) -> None:
        """Cool down between applications, with jitter so the cadence is not fixed."""
        pace = self.pace_for(platform)
        last = self._last_application.get(platform)
        if last is not None:
            target = random.uniform(pace.min_application_gap_s, pace.max_application_gap_s)
            wait = target - (time.monotonic() - last)
            if wait > 0:
                log.info("Pacing: waiting %.0fs before the next %s application", wait, platform)
                await asyncio.sleep(wait)
        self._last_application[platform] = time.monotonic()

    def record_application(self, platform: str) -> None:
        self._daily_counts[(platform, datetime.now(UTC).date())] += 1

    def applications_today(self, platform: str | None = None) -> int:
        today = datetime.now(UTC).date()
        if platform is not None:
            return self._daily_counts[(platform, today)]
        return sum(count for (_, day), count in self._daily_counts.items() if day == today)

    def daily_limit_reached(self, platform: str) -> bool:
        return self.applications_today(platform) >= self.pace_for(platform).daily_limit

    async def on_error(self, platform: str) -> None:
        """Exponential backoff, capped, so a failing platform is not hammered."""
        self._consecutive_errors[platform] += 1
        attempts = self._consecutive_errors[platform]
        delay = min(2.0 ** attempts, 120.0) * random.uniform(0.8, 1.2)
        log.warning(
            "Backing off %s for %.0fs after %s consecutive errors", platform, delay, attempts
        )
        await asyncio.sleep(delay)

    def on_success(self, platform: str) -> None:
        self._consecutive_errors[platform] = 0

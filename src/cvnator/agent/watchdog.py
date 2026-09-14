"""Loop detection and escalation ladder.

Two failure modes kill long autonomous runs, and this module addresses both.

**Spinning.** The agent repeats an action that changes nothing - clicking a disabled
button, refilling a field that a validation rule keeps rejecting. The watchdog
fingerprints each cycle (page identity + action + target) and forces replanning when
the same cycle repeats, then escalates if replanning does not help.

Note what is deliberately *not* treated as spinning: filling six different fields on
one form. Typing into a text input does not change the page structure, so a stall
detector keyed on the page fingerprint alone would fire on every well-behaved form
and drag the vision model in for no reason. Progress is therefore defined as "the
tool reported success", and only consecutive *unsuccessful* cycles on an unchanged
page count toward a stall.

**Grinding.** The agent makes progress but never finishes. A hard step budget per
application bounds that, so a stuck application fails visibly instead of consuming
the session.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from cvnator.domain.actions import AgentDecision, ToolResult
from cvnator.observability import get_logger

log = get_logger(__name__)


class RecoveryStrategy(str, Enum):
    """The escalation ladder, tried in order as attempts fail."""

    RETRY = "retry"
    REINSPECT_DOM = "reinspect_dom"
    REINSPECT_ACCESSIBILITY = "reinspect_accessibility"
    VISUAL_FALLBACK = "visual_fallback"
    REPLAN = "replan"
    ESCALATE_TO_USER = "escalate_to_user"
    ABANDON = "abandon"


LADDER: tuple[RecoveryStrategy, ...] = (
    RecoveryStrategy.RETRY,
    RecoveryStrategy.REINSPECT_DOM,
    RecoveryStrategy.REINSPECT_ACCESSIBILITY,
    RecoveryStrategy.VISUAL_FALLBACK,
    RecoveryStrategy.REPLAN,
    RecoveryStrategy.ESCALATE_TO_USER,
)


def strategy_for_attempt(attempt: int) -> RecoveryStrategy:
    """``attempt`` is zero-based: the first failure gets a plain retry."""
    if attempt < 0:
        attempt = 0
    return LADDER[attempt] if attempt < len(LADDER) else RecoveryStrategy.ABANDON


def cycle_fingerprint(page_fingerprint: str, decision: AgentDecision) -> str:
    payload = "|".join(
        [
            page_fingerprint,
            decision.action.value,
            decision.target.element_id,
            decision.target.selector,
            (decision.value or "")[:60],
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class WatchdogVerdict:
    looping: bool = False
    stalled: bool = False
    exhausted: bool = False
    reason: str = ""
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY

    @property
    def needs_intervention(self) -> bool:
        return self.strategy in {
            RecoveryStrategy.ESCALATE_TO_USER,
            RecoveryStrategy.ABANDON,
        }


@dataclass(slots=True)
class Watchdog:
    loop_threshold: int = 4
    max_steps: int = 60
    stall_threshold: int = 6

    steps: int = 0
    consecutive_failures: int = 0
    consecutive_no_progress: int = 0
    _cycles: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    _fingerprints: deque[str] = field(default_factory=lambda: deque(maxlen=12))

    def reset(self) -> None:
        self.steps = 0
        self.consecutive_failures = 0
        self.consecutive_no_progress = 0
        self._cycles.clear()
        self._fingerprints.clear()

    def record(self, decision: AgentDecision, result: ToolResult) -> WatchdogVerdict:
        """Observe one completed cycle and decide whether to intervene."""
        self.steps += 1
        fingerprint = cycle_fingerprint(result.page_fingerprint, decision)
        self._cycles.append(fingerprint)
        self._fingerprints.append(result.page_fingerprint)

        if result.success:
            self.consecutive_failures = 0
            self.consecutive_no_progress = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_no_progress += 1

        if self.steps >= self.max_steps:
            return WatchdogVerdict(
                exhausted=True,
                reason=f"Step budget exhausted ({self.steps}/{self.max_steps})",
                strategy=RecoveryStrategy.ESCALATE_TO_USER,
            )

        repeats = sum(1 for c in self._cycles if c == fingerprint)
        if repeats >= self.loop_threshold:
            return WatchdogVerdict(
                looping=True,
                reason=(
                    f"Same page, same action, same target repeated {repeats} times "
                    "without progress"
                ),
                strategy=RecoveryStrategy.REPLAN,
            )

        page_frozen = (
            len(self._fingerprints) >= self.stall_threshold
            and len(set(list(self._fingerprints)[-self.stall_threshold :])) == 1
        )
        if page_frozen and self.consecutive_no_progress >= self.stall_threshold:
            return WatchdogVerdict(
                stalled=True,
                reason=(
                    f"The last {self.stall_threshold} actions all failed and the page "
                    "has not changed"
                ),
                strategy=RecoveryStrategy.VISUAL_FALLBACK,
            )

        if self.consecutive_failures >= 3:
            return WatchdogVerdict(
                stalled=True,
                reason=f"{self.consecutive_failures} consecutive tool failures",
                strategy=strategy_for_attempt(self.consecutive_failures),
            )

        return WatchdogVerdict()

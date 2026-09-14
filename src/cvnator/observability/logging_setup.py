"""Logging with domain-specific levels and mandatory secret redaction.

Custom levels sit between INFO and WARNING so they are always visible at INFO,
while still being filterable by the UI:

    DEBUG(10) < INFO(20) < TOOL_CALL(22) < AGENT_DECISION(24)
    < STATE_TRANSITION(26) < USER_ACTION_REQUIRED(28) < WARNING(30)

Nothing that looks like a credential is ever written to a log record. The
redaction happens in a filter attached to the root handlers, so it also covers
third-party libraries and tracebacks.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TOOL_CALL = 22
AGENT_DECISION = 24
STATE_TRANSITION = 26
USER_ACTION_REQUIRED = 28

logging.addLevelName(TOOL_CALL, "TOOL_CALL")
logging.addLevelName(AGENT_DECISION, "AGENT_DECISION")
logging.addLevelName(STATE_TRANSITION, "STATE_TRANSITION")
logging.addLevelName(USER_ACTION_REQUIRED, "USER_ACTION_REQUIRED")

REDACTED = "***REDACTED***"

# Patterns matched case-insensitively against the *formatted* message.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(password|passwd|senha|secret|api[_-]?key|token|authorization|bearer|"
               r"session[_-]?id|cookie|csrf|otp|mfa[_-]?code)\b\s*[:=]\s*(\"[^\"]*\"|'[^']*'|\S+)"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(sk|rk)-[A-Za-z0-9]{12,}"),
)


def redact(text: str) -> str:
    """Strip anything that looks like a credential out of ``text``."""
    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(lambda m: f"{m.group(0).split('=')[0].split(':')[0].rstrip()}={REDACTED}"
                          if ("=" in m.group(0) or ":" in m.group(0)) else REDACTED, out)
    return out


class RedactionFilter(logging.Filter):
    """Applies :func:`redact` to the final message and to structured ``extra`` fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive, malformed % args
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        for key in ("payload", "detail", "value"):
            val = getattr(record, key, None)
            if isinstance(val, str):
                setattr(record, key, redact(val))
        return True


class RingBufferHandler(logging.Handler):
    """Keeps the last N formatted records in memory so the UI can stream them."""

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self.records: deque[dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": self.format(record),
                }
            )
        except Exception:  # pragma: no cover
            self.handleError(record)

    def tail(self, limit: int = 200, levels: Iterable[str] | None = None) -> list[dict[str, Any]]:
        items = list(self.records)
        if levels:
            wanted = {level.upper() for level in levels}
            items = [item for item in items if item["level"] in wanted]
        return items[-limit:]


RING_BUFFER = RingBufferHandler()

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-20s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    redactor = RedactionFilter()

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    stream.addFilter(redactor)

    RING_BUFFER.setFormatter(fmt)
    RING_BUFFER.addFilter(redactor)

    handlers: list[logging.Handler] = [stream, RING_BUFFER]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.addFilter(redactor)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in handlers:
        root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _CONFIGURED = True


class AgentLogger(logging.LoggerAdapter):
    """Logger exposing the domain-specific levels as named methods."""

    def tool_call(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(TOOL_CALL, msg, *args, **kwargs)

    def decision(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(AGENT_DECISION, msg, *args, **kwargs)

    def transition(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(STATE_TRANSITION, msg, *args, **kwargs)

    def needs_user(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(USER_ACTION_REQUIRED, msg, *args, **kwargs)


def get_logger(name: str) -> AgentLogger:
    return AgentLogger(logging.getLogger(name), {})

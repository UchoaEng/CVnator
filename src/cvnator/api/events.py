"""Server-sent events: how the UI watches a long-running session.

SSE rather than WebSockets because the traffic is one-directional (the session talks,
the browser listens), it survives proxies, and it reconnects on its own.

Each subscriber gets its own bounded queue. A slow or abandoned tab drops its oldest
events rather than applying backpressure to the agent - the session must never stall
because a browser window stopped reading.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cvnator.observability import get_logger

log = get_logger(__name__)

QUEUE_MAX = 500


@dataclass
class Event:
    kind: str
    payload: dict[str, Any]
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_sse(self) -> str:
        body = json.dumps({"kind": self.kind, "ts": self.ts, "payload": self.payload})
        return f"event: {self.kind}\ndata: {body}\n\n"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._history: list[Event] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        """Thread-safe publish. Safe to call from the session's worker task."""
        event = Event(kind=kind, payload=payload or {})
        self._history.append(event)
        if len(self._history) > 200:
            del self._history[:-200]

        for queue in list(self._subscribers):
            if queue.full():
                # Drop the oldest so a stalled reader never blocks the producer.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {"kind": e.kind, "ts": e.ts, "payload": e.payload} for e in self._history[-limit:]
        ]

    async def subscribe(self) -> AsyncIterator[str]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=QUEUE_MAX)
        self._subscribers.add(queue)
        try:
            yield Event(kind="connected", payload={"subscribers": len(self._subscribers)}).to_sse()
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield event.to_sse()
                except TimeoutError:
                    # A comment frame keeps proxies and browsers from closing the stream.
                    yield ": keepalive\n\n"
        finally:
            self._subscribers.discard(queue)


bus = EventBus()

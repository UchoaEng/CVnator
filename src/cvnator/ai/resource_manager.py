"""VRAM-aware model residency manager.

Target hardware is a single 12 GB card, so the assumption is that **not every model
fits at once**. Before any generation, the caller acquires the role it needs; the
manager evicts whatever else is resident if the budget would be exceeded, then
loads the requested model.

Eviction order is least-recently-used, with the currently requested model pinned.
A model that has been idle longer than ``model_idle_unload_s`` is dropped even when
there is headroom, so a long browsing stretch does not sit on 7 GB of VRAM.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from cvnator.ai.base import LLMProvider, ModelRole
from cvnator.observability import get_logger

log = get_logger(__name__)

MB = 1024 * 1024

#: Conservative fallback sizes (MB) when the provider cannot report actual usage.
_SIZE_HINTS: tuple[tuple[str, int], ...] = (
    ("70b", 40_000),
    ("32b", 20_000),
    ("27b", 17_000),
    ("14b", 9_500),
    ("13b", 9_000),
    ("12b", 8_000),
    ("9b", 6_800),
    ("8b", 5_600),
    ("7b", 5_000),
    ("4b", 3_400),
    ("3b", 2_600),
    ("1b", 1_200),
)


def estimate_model_mb(model_name: str) -> int:
    lowered = model_name.lower()
    for token, size in _SIZE_HINTS:
        if token in lowered:
            return size
    return 6_000


@dataclass(slots=True)
class ResidentModel:
    name: str
    provider: LLMProvider
    size_mb: int
    last_used: float = field(default_factory=time.monotonic)
    pinned: bool = False


class ResourceManager:
    """Serializes model residency so two roles never fight over the same VRAM."""

    def __init__(self, budget_mb: int, idle_unload_s: float = 180.0) -> None:
        self.budget_mb = budget_mb
        self.idle_unload_s = idle_unload_s
        self._resident: dict[str, ResidentModel] = {}
        self._lock = asyncio.Lock()

    @property
    def used_mb(self) -> int:
        return sum(m.size_mb for m in self._resident.values())

    def snapshot(self) -> dict[str, object]:
        return {
            "budget_mb": self.budget_mb,
            "used_mb": self.used_mb,
            "models": [
                {
                    "name": m.name,
                    "size_mb": m.size_mb,
                    "idle_s": round(time.monotonic() - m.last_used, 1),
                    "pinned": m.pinned,
                }
                for m in self._resident.values()
            ],
        }

    async def _refresh_sizes(self, provider: LLMProvider) -> None:
        """Replace estimates with what the provider actually reports, when it can."""
        try:
            actual = await provider.loaded_models()
        except Exception:  # pragma: no cover - provider without the capability
            return
        for name, size_bytes in actual.items():
            entry = self._resident.get(name)
            if entry is not None and size_bytes > 0:
                entry.size_mb = max(1, size_bytes // MB)

    async def _evict(self, keep: str) -> None:
        """Drop LRU models until the budget is satisfied."""
        candidates = sorted(
            (m for m in self._resident.values() if m.name != keep and not m.pinned),
            key=lambda m: m.last_used,
        )
        for model in candidates:
            if self.used_mb <= self.budget_mb:
                return
            log.info(
                "VRAM budget %s MB exceeded (%s MB used); unloading %s (%s MB)",
                self.budget_mb,
                self.used_mb,
                model.name,
                model.size_mb,
            )
            await model.provider.unload_model(model.name)
            self._resident.pop(model.name, None)

    async def _drop_idle(self, keep: str) -> None:
        now = time.monotonic()
        stale = [
            m
            for m in self._resident.values()
            if m.name != keep and not m.pinned and (now - m.last_used) > self.idle_unload_s
        ]
        for model in stale:
            log.info("Unloading idle model %s after %.0fs", model.name, now - model.last_used)
            await model.provider.unload_model(model.name)
            self._resident.pop(model.name, None)

    async def acquire(self, provider: LLMProvider, model: str, role: ModelRole) -> None:
        """Ensure ``model`` is resident and within budget before generation."""
        async with self._lock:
            await self._drop_idle(keep=model)
            existing = self._resident.get(model)
            if existing is not None:
                existing.last_used = time.monotonic()
                await self._refresh_sizes(provider)
                await self._evict(keep=model)
                return

            size_mb = estimate_model_mb(model)
            projected = self.used_mb + size_mb
            if projected > self.budget_mb:
                log.info(
                    "Loading %s for role %s needs ~%s MB; making room (%s/%s MB in use)",
                    model,
                    role.value,
                    size_mb,
                    self.used_mb,
                    self.budget_mb,
                )
                self._resident[model] = ResidentModel(model, provider, size_mb, pinned=True)
                try:
                    await self._evict(keep=model)
                finally:
                    self._resident[model].pinned = False
            else:
                self._resident[model] = ResidentModel(model, provider, size_mb)

            await provider.load_model(model)
            await self._refresh_sizes(provider)
            log.info(
                "Model %s resident for role %s (%s/%s MB)",
                model,
                role.value,
                self.used_mb,
                self.budget_mb,
            )

    async def release_all(self) -> None:
        """Unload everything. Called on session stop and application shutdown."""
        async with self._lock:
            for model in list(self._resident.values()):
                await model.provider.unload_model(model.name)
            self._resident.clear()

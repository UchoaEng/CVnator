"""The AI runtime: which models exist, which ones fit, and what to do when they don't.

Models are replaceable components, not a fixed dependency. This module owns that
replaceability: it reads what is installed, estimates what each variant will cost,
refuses configurations that cannot load, recommends one that fits the machine in
front of it, and walks a recovery ladder when the card runs out of memory.

Two rules shape everything here:

* **Estimates are labelled as estimates.** Real memory use depends on the backend,
  the context, the KV cache and what else holds the card. A number presented as
  exact would be a lie the user only discovers at load time.
* **A model that cannot do the job is not offered for it.** Asking a text-only model
  to read a screenshot fails at the worst possible moment; capabilities are checked
  when the choice is made, not when the work starts.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

from cvnator.ai.base import LLMProvider, ModelRole, ProviderError
from cvnator.ai.hardware import (
    TIER_CPU_ONLY,
    TIER_HIGH,
    TIER_LOW,
    TIER_RECOMMENDED,
    TIER_STANDARD,
    HardwareProfile,
)
from cvnator.observability import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------- quantization

#: Relative weight cost per parameter, in bytes, for each quantization family.
#: These are the accepted approximate figures for GGUF k-quants; they are used for
#: *ordering and rough sizing*, never presented as the exact footprint.
_BITS_PER_WEIGHT = {
    "F32": 4.0,
    "FP32": 4.0,
    "F16": 2.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "Q8": 1.06,
    "Q6": 0.82,
    "Q5": 0.71,
    "Q4": 0.58,
    "Q3": 0.47,
    "Q2": 0.36,
    "INT8": 1.06,
    "INT4": 0.58,
}

#: How a variant is described to someone who does not want to learn what K_M means.
#: Quality is relative *to the same model*: a Q4 of a larger model may well beat a
#: Q8 of a smaller one, which is why the UI must never sort on this alone.
_QUALITY_BY_FAMILY = {
    "F32": "Maximum",
    "FP32": "Maximum",
    "F16": "Very high",
    "FP16": "Very high",
    "BF16": "Very high",
    "Q8": "Very high",
    "Q6": "High",
    "Q5": "High",
    "Q4": "Good",
    "Q3": "Reduced",
    "Q2": "Low",
    "INT8": "Very high",
    "INT4": "Good",
}

_MEMORY_BY_FAMILY = {
    "F32": "Very high",
    "FP32": "Very high",
    "F16": "High",
    "FP16": "High",
    "BF16": "High",
    "Q8": "High",
    "Q6": "Medium",
    "Q5": "Medium",
    "Q4": "Low",
    "Q3": "Very low",
    "Q2": "Very low",
    "INT8": "High",
    "INT4": "Low",
}

#: Preference order when the recommender has a free choice, lightest-acceptable first
#: per tier. Q4_K_M is the usual sweet spot and leads on constrained hardware.
_QUANT_PREFERENCE = {
    TIER_CPU_ONLY: ["Q4", "Q3", "Q5"],
    TIER_LOW: ["Q4", "Q3", "Q5"],
    TIER_STANDARD: ["Q4", "Q5", "Q6"],
    TIER_RECOMMENDED: ["Q5", "Q4", "Q6"],
    TIER_HIGH: ["Q6", "Q5", "Q8"],
}

_QUANT_RE = re.compile(r"(F32|FP32|F16|FP16|BF16|INT8|INT4|Q\d)", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class Quantization:
    """A quantization variant, described in both technical and human terms."""

    label: str
    family: str
    bytes_per_weight: float
    quality: str
    memory: str

    @property
    def is_unknown(self) -> bool:
        return self.family == "UNKNOWN"


def parse_quantization(label: str) -> Quantization:
    """Turn ``Q4_K_M`` / ``Q5_K_S`` / ``F16`` / ``AWQ`` into something comparable.

    An unrecognised label is *not* an error: new formats appear constantly. It is
    reported as unknown, sized pessimistically, and still usable.
    """
    raw = (label or "").strip()
    match = _QUANT_RE.search(raw)
    if match is None:
        return Quantization(
            label=raw or "unknown",
            family="UNKNOWN",
            # Pessimistic: assume FP16-like weight cost so an unknown format is never
            # recommended into a card it cannot fit.
            bytes_per_weight=2.0,
            quality="Unknown",
            memory="Unknown",
        )
    family = match.group(1).upper()
    return Quantization(
        label=raw,
        family=family,
        bytes_per_weight=_BITS_PER_WEIGHT.get(family, 2.0),
        quality=_QUALITY_BY_FAMILY.get(family, "Unknown"),
        memory=_MEMORY_BY_FAMILY.get(family, "Unknown"),
    )


def _parse_parameter_count(text: str) -> float:
    """``"9.7B"`` -> 9.7e9. Returns 0.0 when the string says nothing useful."""
    if not text:
        return 0.0
    match = re.match(r"\s*([\d.]+)\s*([BMK])?", str(text), re.IGNORECASE)
    if not match:
        return 0.0
    try:
        value = float(match.group(1))
    except ValueError:
        return 0.0
    suffix = (match.group(2) or "").upper()
    return value * {"B": 1e9, "M": 1e6, "K": 1e3}.get(suffix, 1.0)


# --------------------------------------------------------------------------- models


@dataclass(slots=True)
class ModelInfo:
    """One installed model, as the runtime reports it."""

    name: str
    backend: str = "ollama"
    family: str = ""
    parameter_text: str = ""
    parameter_count: float = 0.0
    fmt: str = ""
    quantization: Quantization = field(
        default_factory=lambda: parse_quantization("")
    )
    file_size_bytes: int = 0
    context_length: int = 0
    capabilities: list[str] = field(default_factory=list)

    @property
    def supports_vision(self) -> bool:
        return "vision" in self.capabilities

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def supports_embedding(self) -> bool:
        return "embedding" in self.capabilities or "embed" in self.capabilities

    @property
    def file_size_mb(self) -> int:
        return int(self.file_size_bytes / (1024 * 1024))

    def supports_role(self, role: ModelRole) -> bool:
        """Whether this model can do the job at all.

        Reasoning, resume and translation are all text generation; only vision and
        embedding need a capability the model may simply not have.
        """
        if role is ModelRole.VISION:
            return self.supports_vision
        if role is ModelRole.EMBEDDING:
            return self.supports_embedding or bool(self.capabilities) is False
        return "completion" in self.capabilities or not self.capabilities

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quantization"] = asdict(self.quantization)
        data["file_size_mb"] = self.file_size_mb
        data["supports_vision"] = self.supports_vision
        data["supports_tools"] = self.supports_tools
        return data


def model_from_ollama_tag(entry: dict[str, Any]) -> ModelInfo:
    """Build a :class:`ModelInfo` from one entry of Ollama's ``/api/tags``."""
    details = entry.get("details") or {}
    return ModelInfo(
        name=str(entry.get("name") or entry.get("model") or ""),
        backend="ollama",
        family=str(details.get("family") or ""),
        parameter_text=str(details.get("parameter_size") or ""),
        parameter_count=_parse_parameter_count(details.get("parameter_size", "")),
        fmt=str(details.get("format") or ""),
        quantization=parse_quantization(str(details.get("quantization_level") or "")),
        file_size_bytes=int(entry.get("size") or 0),
        context_length=int(details.get("context_length") or 0),
        capabilities=[str(c) for c in (entry.get("capabilities") or [])],
    )


# ---------------------------------------------------------------------- estimation

#: Rough KV-cache cost in MB per 1k tokens of context, per billion parameters, with
#: a 16-bit cache. Calibrated against a well-known 8B model with grouped-query
#: attention (32 layers, 8 KV heads, 128 head dim), which needs about 1 GB at 8k
#: tokens. Architectures without GQA cost several times more, which is one reason
#: the result is reported as a range and labelled an estimate.
_KV_MB_PER_1K_PER_B = 16.8

#: Fixed overhead for the runtime, CUDA context and compute buffers.
_RUNTIME_OVERHEAD_MB = 600


@dataclass(slots=True)
class MemoryEstimate:
    """An estimate with its uncertainty attached, because the range is the point."""

    low_mb: int
    expected_mb: int
    high_mb: int
    weights_mb: int
    kv_cache_mb: int
    overhead_mb: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_memory(model: ModelInfo, context_tokens: int = 8192) -> MemoryEstimate:
    """Estimate what loading this model at this context will cost.

    Prefers the real file size over a parameter-count calculation: the file is what
    gets read into memory, and it already reflects the quantization actually used.
    """
    if model.file_size_mb:
        weights_mb = model.file_size_mb
    elif model.parameter_count:
        weights_mb = int(
            model.parameter_count * model.quantization.bytes_per_weight / (1024 * 1024)
        )
    else:
        weights_mb = 0

    billions = (model.parameter_count / 1e9) if model.parameter_count else 0.0
    if not billions and weights_mb:
        # Work backwards from the file when the runtime did not report a size.
        billions = weights_mb * 1024 * 1024 / max(model.quantization.bytes_per_weight, 0.1) / 1e9
    kv_mb = int((context_tokens / 1000) * billions * _KV_MB_PER_1K_PER_B)

    expected = weights_mb + kv_mb + _RUNTIME_OVERHEAD_MB
    return MemoryEstimate(
        # The band is deliberately asymmetric: underestimating causes an OOM, while
        # overestimating only causes a more conservative recommendation.
        low_mb=int(expected * 0.9),
        expected_mb=int(expected),
        high_mb=int(expected * 1.25),
        weights_mb=weights_mb,
        kv_cache_mb=kv_mb,
        overhead_mb=_RUNTIME_OVERHEAD_MB,
    )


# -------------------------------------------------------------------- compatibility

VERDICT_OK = "ok"
VERDICT_WARN = "warn"
VERDICT_BLOCK = "block"


@dataclass(slots=True)
class Compatibility:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    estimate: MemoryEstimate | None = None

    @property
    def usable(self) -> bool:
        return self.verdict != VERDICT_BLOCK

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": self.reasons,
            "suggestions": self.suggestions,
            "estimate": self.estimate.as_dict() if self.estimate else None,
            "usable": self.usable,
        }


def check_compatibility(
    model: ModelInfo,
    role: ModelRole,
    hardware: HardwareProfile,
    *,
    context_tokens: int = 8192,
    cpu_offload: bool = False,
) -> Compatibility:
    """Can this model do this job on this machine, and at what risk.

    A missing capability blocks. Insufficient memory warns when CPU offload can
    rescue it and blocks when nothing can - the point is to fail here, in a settings
    screen, rather than four hours into an autonomous session.
    """
    reasons: list[str] = []
    suggestions: list[str] = []

    if not model.supports_role(role):
        missing = "image input" if role is ModelRole.VISION else "embeddings"
        return Compatibility(
            verdict=VERDICT_BLOCK,
            reasons=[f"{model.name} does not support {missing}."],
            suggestions=[f"Choose a model that lists {missing} among its capabilities."],
        )

    if model.context_length and context_tokens > model.context_length:
        reasons.append(
            f"The requested context ({context_tokens} tokens) is larger than this "
            f"model supports ({model.context_length})."
        )
        suggestions.append(f"Reduce the context to {model.context_length} tokens or fewer.")
        return Compatibility(verdict=VERDICT_BLOCK, reasons=reasons, suggestions=suggestions)

    estimate = estimate_memory(model, context_tokens)

    if not hardware.gpu.detected:
        # CPU-only is supported, not blocked: slower, but the product still works.
        if hardware.ram_total_mb and estimate.high_mb > hardware.ram_total_mb:
            return Compatibility(
                verdict=VERDICT_BLOCK,
                reasons=[
                    f"No GPU was detected and this model needs roughly "
                    f"{estimate.expected_mb} MB, more than the {hardware.ram_total_mb} MB "
                    f"of system memory."
                ],
                suggestions=["Choose a smaller model or a more compact variant."],
                estimate=estimate,
            )
        return Compatibility(
            verdict=VERDICT_WARN,
            reasons=["No GPU was detected; this model will run on the CPU and be slow."],
            suggestions=["A smaller model will respond noticeably faster on this machine."],
            estimate=estimate,
        )

    usable = hardware.usable_vram_mb()
    if estimate.high_mb <= usable:
        return Compatibility(verdict=VERDICT_OK, estimate=estimate)

    if estimate.expected_mb <= usable:
        return Compatibility(
            verdict=VERDICT_WARN,
            reasons=[
                f"This should fit, but with little room to spare: roughly "
                f"{estimate.expected_mb} MB against {usable} MB usable."
            ],
            suggestions=["Close other GPU-heavy applications, or reduce the context size."],
            estimate=estimate,
        )

    reasons.append(
        f"This configuration is likely to exceed your graphics memory: roughly "
        f"{estimate.expected_mb} MB needed against {usable} MB usable."
    )
    if cpu_offload:
        suggestions.append("Part of the model will run on the CPU, which is slower.")
        return Compatibility(verdict=VERDICT_WARN, reasons=reasons, suggestions=suggestions,
                             estimate=estimate)

    suggestions.append("Use a more compact variant (for example Q4_K_M) or a smaller model.")
    suggestions.append("Or enable CPU offload, which trades speed for memory.")
    verdict = VERDICT_BLOCK if estimate.low_mb > usable * 1.5 else VERDICT_WARN
    return Compatibility(
        verdict=verdict, reasons=reasons, suggestions=suggestions, estimate=estimate
    )


# ------------------------------------------------------------------ recommendation

#: Context sizes per tier. Context is the cheapest thing to give up on a small card
#: and the first thing the recommender reduces.
_CONTEXT_BY_TIER = {
    TIER_CPU_ONLY: 4096,
    TIER_LOW: 4096,
    TIER_STANDARD: 8192,
    TIER_RECOMMENDED: 16384,
    TIER_HIGH: 32768,
}

PROFILE_ECO = "eco"
PROFILE_BALANCED = "balanced"
PROFILE_QUALITY = "quality"
PROFILE_MAXIMUM = "maximum"
PROFILE_CUSTOM = "custom"

_PROFILE_BY_TIER = {
    TIER_CPU_ONLY: PROFILE_ECO,
    TIER_LOW: PROFILE_ECO,
    TIER_STANDARD: PROFILE_BALANCED,
    TIER_RECOMMENDED: PROFILE_BALANCED,
    TIER_HIGH: PROFILE_QUALITY,
}


@dataclass(slots=True)
class RoleConfig:
    """What one role runs on."""

    model: str = ""
    quantization: str = ""
    context_tokens: int = 8192
    cpu_offload: bool = False
    fallback_model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AIConfiguration:
    """The complete AI runtime configuration, as persisted and as shown in Settings."""

    profile: str = PROFILE_BALANCED
    low_vram_mode: bool = False
    auto_fallback: bool = True
    roles: dict[str, RoleConfig] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "low_vram_mode": self.low_vram_mode,
            "auto_fallback": self.auto_fallback,
            "roles": {role: config.as_dict() for role, config in self.roles.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AIConfiguration:
        roles = {
            str(role): RoleConfig(**{**RoleConfig().as_dict(), **dict(value)})
            for role, value in (data.get("roles") or {}).items()
        }
        return cls(
            profile=str(data.get("profile") or PROFILE_BALANCED),
            low_vram_mode=bool(data.get("low_vram_mode", False)),
            auto_fallback=bool(data.get("auto_fallback", True)),
            roles=roles,
        )


def _score_for_role(model: ModelInfo, role: ModelRole, tier: str) -> tuple:
    """Sort key for picking a model. Bigger is not automatically better.

    Preference order: the quantization families this tier likes, then more
    parameters, then tool support (the agent loop needs structured output).
    """
    preference = _QUANT_PREFERENCE.get(tier, ["Q4"])
    try:
        quant_rank = preference.index(model.quantization.family)
    except ValueError:
        quant_rank = len(preference)
    return (quant_rank, -model.parameter_count, 0 if model.supports_tools else 1)


def recommend_configuration(
    hardware: HardwareProfile, installed: list[ModelInfo]
) -> AIConfiguration:
    """Choose a configuration that fits this machine, from what is actually installed.

    Never recommends a model that is not present: a recommendation the user cannot
    act on is worse than none. Roles with no compatible model are left empty and the
    UI says so.
    """
    tier = hardware.tier
    context = _CONTEXT_BY_TIER.get(tier, 8192)
    config = AIConfiguration(
        profile=_PROFILE_BY_TIER.get(tier, PROFILE_BALANCED),
        low_vram_mode=hardware.low_vram,
        auto_fallback=True,
    )

    for role in (ModelRole.REASONING, ModelRole.RESUME, ModelRole.TRANSLATION):
        candidates = [
            model
            for model in installed
            if model.supports_role(role)
            and check_compatibility(
                model, role, hardware, context_tokens=context
            ).verdict
            != VERDICT_BLOCK
        ]
        if not candidates:
            config.roles[role.value] = RoleConfig(context_tokens=context)
            continue
        ranked = sorted(candidates, key=lambda m: _score_for_role(m, role, tier))
        chosen = ranked[0]
        # The fallback is the lightest compatible model, so that when the primary
        # cannot load there is something that certainly can.
        lightest = min(candidates, key=lambda m: m.file_size_mb or int(m.parameter_count))
        config.roles[role.value] = RoleConfig(
            model=chosen.name,
            quantization=chosen.quantization.label,
            context_tokens=min(context, chosen.context_length or context),
            cpu_offload=hardware.low_vram,
            fallback_model=lightest.name if lightest.name != chosen.name else "",
        )

    return config


# ------------------------------------------------------------------- model manager


class ModelUnavailable(ProviderError):
    """Raised when no configuration of this role can be made to load."""


#: Substrings that identify an out-of-memory failure across backends. Matching on the
#: message is unpleasant but it is what the runtimes give us.
_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda error",
    "cublas",
    "failed to allocate",
    "insufficient memory",
    "vram",
)


def is_out_of_memory(error: BaseException) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


@dataclass(slots=True)
class LoadedModel:
    name: str
    role: str
    loaded_at: float
    estimate_mb: int


class ModelManager:
    """Owns the local model runtime: inventory, fit, loading and recovery.

    Deliberately does not import the database. Persistence is injected, so the
    manager is testable without a session and reusable from a CLI.
    """

    def __init__(
        self,
        provider: LLMProvider,
        hardware: HardwareProfile,
        *,
        load_config: Callable[[], dict[str, Any]] | None = None,
        save_config: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.provider = provider
        self.hardware = hardware
        self._load_config = load_config
        self._save_config = save_config
        self._installed: list[ModelInfo] = []
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        #: Counters the AI observability section reports. No prompt content, ever.
        self.metrics: dict[str, int | float] = {
            "load_count": 0,
            "unload_count": 0,
            "oom_count": 0,
            "fallback_count": 0,
            "last_load_seconds": 0.0,
        }

    # ------------------------------------------------------------------ inventory

    async def refresh_installed(self) -> list[ModelInfo]:
        """Ask the backend what is installed. An unreachable backend yields []."""
        lister = getattr(self.provider, "list_model_details", None)
        if lister is None:
            log.warning("Provider %s cannot list models", type(self.provider).__name__)
            self._installed = []
            return []
        try:
            entries = await lister()
        except Exception as exc:  # backend down is a state, not a crash
            log.warning("Could not list installed models: %s", exc)
            self._installed = []
            return []
        self._installed = [model_from_ollama_tag(entry) for entry in entries]
        log.info("Found %d installed model(s)", len(self._installed))
        return self._installed

    @property
    def installed(self) -> list[ModelInfo]:
        return list(self._installed)

    def find(self, name: str) -> ModelInfo | None:
        for model in self._installed:
            if model.name == name:
                return model
        # Ollama accepts "qwen3.5:9b" and "qwen3.5" for the same tag.
        for model in self._installed:
            if model.name.split(":")[0] == name.split(":")[0]:
                return model
        return None

    def models_for_role(self, role: ModelRole) -> list[ModelInfo]:
        return [model for model in self._installed if model.supports_role(role)]

    # ------------------------------------------------------------- configuration

    def load_configuration(self) -> AIConfiguration:
        """The saved configuration, or one recommended for this machine."""
        if self._load_config is not None:
            stored = self._load_config()
            if stored:
                return AIConfiguration.from_dict(stored)
        return recommend_configuration(self.hardware, self._installed)

    def save_configuration(self, config: AIConfiguration) -> None:
        if self._save_config is not None:
            self._save_config(config.as_dict())

    def recommend(self) -> AIConfiguration:
        return recommend_configuration(self.hardware, self._installed)

    def validate_configuration(self, config: AIConfiguration) -> dict[str, Compatibility]:
        """Pre-flight for the AI runtime: every configured role checked in one pass."""
        results: dict[str, Compatibility] = {}
        for role_name, role_config in config.roles.items():
            try:
                role = ModelRole(role_name)
            except ValueError:
                continue
            if not role_config.model:
                results[role_name] = Compatibility(
                    verdict=VERDICT_BLOCK,
                    reasons=["No model is configured for this role."],
                    suggestions=["Choose a model, or accept the recommended configuration."],
                )
                continue
            model = self.find(role_config.model)
            if model is None:
                results[role_name] = Compatibility(
                    verdict=VERDICT_BLOCK,
                    reasons=[f"{role_config.model} is not installed."],
                    suggestions=["Install it, or choose one of the installed models."],
                )
                continue
            results[role_name] = check_compatibility(
                model,
                role,
                self.hardware,
                context_tokens=role_config.context_tokens,
                cpu_offload=role_config.cpu_offload,
            )
        return results

    # -------------------------------------------------------------- load / unload

    async def load(self, name: str, role: ModelRole) -> ModelInfo:
        """Load one model, evicting nothing. Raises :class:`ModelUnavailable`."""
        model = self.find(name)
        if model is None:
            raise ModelUnavailable(f"{name} is not installed")
        started = time.monotonic()
        async with self._lock:
            await self.provider.load_model(name)
        elapsed = time.monotonic() - started
        self.metrics["load_count"] = int(self.metrics["load_count"]) + 1
        self.metrics["last_load_seconds"] = round(elapsed, 2)
        self._loaded[name] = LoadedModel(
            name=name,
            role=role.value,
            loaded_at=time.time(),
            estimate_mb=estimate_memory(model).expected_mb,
        )
        log.info("Loaded %s for %s in %.1fs", name, role.value, elapsed)
        return model

    async def unload(self, name: str) -> None:
        async with self._lock:
            await self.provider.unload_model(name)
        self._loaded.pop(name, None)
        self.metrics["unload_count"] = int(self.metrics["unload_count"]) + 1
        log.info("Unloaded %s", name)

    async def unload_all_except(self, keep: str = "") -> int:
        """Free the card. Returns how many models were unloaded."""
        freed = 0
        for name in list(self._loaded):
            if name == keep:
                continue
            await self.unload(name)
            freed += 1
        return freed

    async def swap(self, old: str, new: str, role: ModelRole) -> ModelInfo:
        """Replace one resident model with another without restarting anything."""
        if old and old != new:
            await self.unload(old)
        return await self.load(new, role)

    def loaded(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self._loaded.values()]

    # ------------------------------------------------------------- OOM recovery

    async def run_with_recovery(
        self,
        operation: Callable[[RoleConfig], Awaitable[T]],
        role: ModelRole,
        config: AIConfiguration,
    ) -> T:
        """Run one inference, walking the recovery ladder if the card runs out.

        The ladder, in order: free the card, halve the context, use the fallback
        model. Each rung is tried once. Only when all of them fail does this raise,
        and the caller turns that into a user action rather than a lost application.
        """
        role_config = config.roles.get(role.value) or RoleConfig()
        attempts: list[tuple[str, RoleConfig]] = [("as configured", role_config)]

        reduced = RoleConfig(**role_config.as_dict())
        reduced.context_tokens = max(2048, role_config.context_tokens // 2)
        attempts.append(("with a smaller context", reduced))

        if config.auto_fallback and role_config.fallback_model:
            lighter = RoleConfig(**reduced.as_dict())
            lighter.model = role_config.fallback_model
            lighter.fallback_model = ""
            attempts.append(("using the lighter fallback model", lighter))

        last_error: BaseException | None = None
        for index, (description, attempt) in enumerate(attempts):
            try:
                return await operation(attempt)
            except Exception as exc:
                if not is_out_of_memory(exc):
                    raise
                last_error = exc
                self.metrics["oom_count"] = int(self.metrics["oom_count"]) + 1
                log.warning(
                    "Out of memory %s (attempt %d/%d): %s",
                    description, index + 1, len(attempts), exc,
                )
                # Free whatever is resident before the next rung; the model we are
                # about to use is the only one that should hold the card.
                await self.unload_all_except(attempt.model)
                if index + 1 < len(attempts) and attempts[index + 1][1].model != attempt.model:
                    self.metrics["fallback_count"] = int(self.metrics["fallback_count"]) + 1

        raise ModelUnavailable(
            "The AI engine ran out of graphics memory and no lighter configuration "
            "worked. Choose a smaller model or a more compact variant."
        ) from last_error

    # ----------------------------------------------------------------- model test

    async def test_model(self, name: str, role: ModelRole = ModelRole.REASONING) -> dict[str, Any]:
        """Load the model, run one tiny completion, and report what it cost.

        Measured, not estimated: this is the answer to "will this actually work".
        """
        from cvnator.ai.base import ChatMessage

        model = self.find(name)
        if model is None:
            return {"ok": False, "error": f"{name} is not installed"}

        started = time.monotonic()
        try:
            await self.load(name, role)
            load_seconds = time.monotonic() - started

            inference_started = time.monotonic()
            result = await self.provider.chat(
                model=name,
                messages=[ChatMessage(role="user", content="Reply with the word: ready")],
                temperature=0.0,
                max_tokens=16,
            )
            inference_seconds = time.monotonic() - inference_started
        except Exception as exc:
            log.warning("Model test failed for %s: %s", name, exc)
            return {
                "ok": False,
                "model": name,
                "error": str(exc)[:300],
                "out_of_memory": is_out_of_memory(exc),
            }

        tokens = result.completion_tokens or len(result.text.split())
        peak_mb = 0
        try:
            resident = await self.provider.loaded_models()
            # The provider reports bytes; the UI speaks in megabytes.
            peak_mb = int(resident.get(name, 0) / (1024 * 1024))
        except Exception:  # pragma: no cover - reporting is best effort
            peak_mb = 0

        return {
            "ok": True,
            "model": name,
            "load_seconds": round(load_seconds, 2),
            "inference_seconds": round(inference_seconds, 2),
            "tokens_per_second": round(tokens / inference_seconds, 1) if inference_seconds else 0,
            "resident_mb": peak_mb,
            "estimated_mb": estimate_memory(model).expected_mb,
            "quantization": model.quantization.label,
        }

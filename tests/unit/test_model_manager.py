"""The AI runtime has to behave on hardware we do not have.

Every machine in these tests is fictional on purpose: a 6 GB card, a 24 GB card and
a machine with no GPU at all are the cases that cannot be checked by running the
product here. What they pin is the promise that a smaller card gets a working
configuration rather than a refusal.
"""

from __future__ import annotations

from typing import Any

import pytest

from cvnator.ai.base import ChatMessage, CompletionResult, LLMProvider, ModelRole
from cvnator.ai.hardware import (
    TIER_CPU_ONLY,
    TIER_HIGH,
    TIER_LOW,
    TIER_RECOMMENDED,
    GPUInfo,
    HardwareProfile,
)
from cvnator.ai.model_manager import (
    VERDICT_BLOCK,
    VERDICT_OK,
    VERDICT_WARN,
    AIConfiguration,
    ModelManager,
    ModelUnavailable,
    RoleConfig,
    check_compatibility,
    estimate_memory,
    is_out_of_memory,
    model_from_ollama_tag,
    parse_quantization,
    recommend_configuration,
)


def _tag(
    name: str,
    *,
    params: str = "9.7B",
    quant: str = "Q4_K_M",
    size_gb: float = 6.2,
    context: int = 32768,
    capabilities: tuple[str, ...] = ("completion", "tools"),
) -> dict[str, Any]:
    return {
        "name": name,
        "size": int(size_gb * 1024 * 1024 * 1024),
        "details": {
            "format": "gguf",
            "family": name.split(":")[0],
            "parameter_size": params,
            "quantization_level": quant,
            "context_length": context,
        },
        "capabilities": list(capabilities),
    }


def _machine(vram_mb: int, ram_mb: int = 32000) -> HardwareProfile:
    gpu = GPUInfo(
        name="Test GPU", vram_total_mb=vram_mb, vram_free_mb=vram_mb, vendor="NVIDIA"
    )
    return HardwareProfile(
        gpu=gpu if vram_mb else GPUInfo(),
        ram_total_mb=ram_mb,
        ram_available_mb=ram_mb // 2,
        cpu_cores=8,
    )


class _FakeProvider(LLMProvider):
    """A provider that records what was asked of it and can be told to fail."""

    def __init__(self, tags: list[dict[str, Any]] | None = None) -> None:
        self.tags = tags or []
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self.calls: list[str] = []
        self.fail_with: BaseException | None = None

    async def list_model_details(self) -> list[dict[str, Any]]:
        return self.tags

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> CompletionResult:
        self.calls.append(str(kwargs.get("model", "")))
        if self.fail_with is not None:
            raise self.fail_with
        return CompletionResult(text="ready", model=str(kwargs.get("model", "")), completion_tokens=2)

    async def is_available(self) -> bool:
        return True

    async def load_model(self, model: str) -> None:
        self.loaded.append(model)

    async def unload_model(self, model: str) -> None:
        self.unloaded.append(model)

    async def loaded_models(self) -> dict[str, int]:
        return {name: 6 * 1024 * 1024 * 1024 for name in self.loaded}


# --------------------------------------------------------------------- quantization


def test_quantization_labels_are_parsed_into_comparable_families() -> None:
    assert parse_quantization("Q4_K_M").family == "Q4"
    assert parse_quantization("Q5_K_S").family == "Q5"
    assert parse_quantization("F16").family == "F16"
    assert parse_quantization("Q8_0").quality == "Very high"
    assert parse_quantization("Q3_K_M").memory == "Very low"


def test_an_unknown_quantization_is_sized_pessimistically_not_rejected() -> None:
    """A format we have never seen must not be recommended into a card it overflows."""
    unknown = parse_quantization("AWQ-marlin")

    assert unknown.is_unknown
    assert unknown.bytes_per_weight == 2.0  # as costly as FP16 until proven otherwise
    assert unknown.quality == "Unknown"


# ------------------------------------------------------------------------ inventory


def test_ollama_metadata_becomes_a_usable_model_record() -> None:
    model = model_from_ollama_tag(
        _tag("qwen3.5:9b", capabilities=("vision", "completion", "tools"))
    )

    assert model.parameter_count == pytest.approx(9.7e9)
    assert model.quantization.family == "Q4"
    assert model.supports_vision and model.supports_tools
    assert model.file_size_mb > 6000


def test_a_text_only_model_is_never_offered_for_vision_work() -> None:
    text_only = model_from_ollama_tag(_tag("text:7b", capabilities=("completion",)))

    assert text_only.supports_role(ModelRole.REASONING)
    assert not text_only.supports_role(ModelRole.VISION)


# ---------------------------------------------------------------------- estimation


def test_context_size_is_part_of_the_estimate_not_an_afterthought() -> None:
    model = model_from_ollama_tag(_tag("m:9b", size_gb=6.0))

    small = estimate_memory(model, context_tokens=4096)
    large = estimate_memory(model, context_tokens=32768)

    assert large.expected_mb > small.expected_mb
    assert large.kv_cache_mb > small.kv_cache_mb
    # The band must bracket the expectation, since that is what the UI shows.
    assert small.low_mb < small.expected_mb < small.high_mb


# --------------------------------------------------------------------- compatibility


def test_a_model_that_cannot_see_is_blocked_from_the_vision_role() -> None:
    model = model_from_ollama_tag(_tag("text:7b", capabilities=("completion",)))

    verdict = check_compatibility(model, ModelRole.VISION, _machine(12000))

    assert verdict.verdict == VERDICT_BLOCK
    assert not verdict.usable
    assert "image input" in verdict.reasons[0]


def test_a_comfortable_fit_is_reported_as_ok() -> None:
    model = model_from_ollama_tag(_tag("small:4b", params="4B", size_gb=2.5))

    verdict = check_compatibility(model, ModelRole.REASONING, _machine(24000))

    assert verdict.verdict == VERDICT_OK


def test_a_large_model_on_a_small_card_is_refused_with_a_way_forward() -> None:
    model = model_from_ollama_tag(_tag("big:70b", params="70B", size_gb=40))

    verdict = check_compatibility(model, ModelRole.REASONING, _machine(6000))

    assert verdict.verdict == VERDICT_BLOCK
    # ADR-020: a refusal must name the next action.
    assert verdict.suggestions
    assert any("compact" in s or "smaller" in s for s in verdict.suggestions)


def test_cpu_offload_downgrades_a_block_to_a_warning() -> None:
    """Offload trades speed for memory; it must not be silently ignored."""
    model = model_from_ollama_tag(_tag("mid:14b", params="14B", size_gb=9))

    strict = check_compatibility(model, ModelRole.REASONING, _machine(6000))
    offloaded = check_compatibility(
        model, ModelRole.REASONING, _machine(6000), cpu_offload=True
    )

    assert strict.verdict == VERDICT_BLOCK
    assert offloaded.verdict == VERDICT_WARN
    assert offloaded.usable


def test_a_machine_with_no_gpu_still_gets_a_working_configuration() -> None:
    model = model_from_ollama_tag(_tag("small:4b", params="4B", size_gb=2.5))

    verdict = check_compatibility(model, ModelRole.REASONING, _machine(0, ram_mb=16000))

    assert verdict.verdict == VERDICT_WARN  # slow, not impossible
    assert verdict.usable


def test_a_context_larger_than_the_model_supports_is_blocked() -> None:
    model = model_from_ollama_tag(_tag("m:9b", context=8192))

    verdict = check_compatibility(
        model, ModelRole.REASONING, _machine(24000), context_tokens=32768
    )

    assert verdict.verdict == VERDICT_BLOCK
    assert "8192" in verdict.suggestions[0]


# -------------------------------------------------------------------- recommendation


def test_hardware_tiers_follow_the_card() -> None:
    assert _machine(0).tier == TIER_CPU_ONLY
    assert _machine(6000).tier == TIER_LOW
    assert _machine(12227).tier == TIER_RECOMMENDED
    assert _machine(24000).tier == TIER_HIGH
    assert _machine(6000).low_vram
    assert not _machine(12227).low_vram


def test_a_small_card_is_recommended_the_model_that_fits_it() -> None:
    installed = [
        model_from_ollama_tag(_tag("big:14b", params="14B", size_gb=9.5)),
        model_from_ollama_tag(_tag("small:4b", params="4B", size_gb=2.4)),
    ]

    config = recommend_configuration(_machine(6000), installed)

    assert config.roles["reasoning"].model == "small:4b"
    assert config.low_vram_mode
    assert config.roles["reasoning"].cpu_offload
    assert config.roles["reasoning"].context_tokens <= 4096


def test_a_large_card_is_not_forced_onto_the_small_model() -> None:
    installed = [
        model_from_ollama_tag(_tag("big:14b", params="14B", size_gb=9.5)),
        model_from_ollama_tag(_tag("small:4b", params="4B", size_gb=2.4)),
    ]

    config = recommend_configuration(_machine(24000), installed)

    assert config.roles["reasoning"].model == "big:14b"
    assert not config.low_vram_mode
    # The lighter model becomes the fallback rather than being discarded.
    assert config.roles["reasoning"].fallback_model == "small:4b"


def test_a_role_with_no_compatible_model_is_left_empty_rather_than_guessed() -> None:
    config = recommend_configuration(_machine(4000), [])

    assert config.roles["reasoning"].model == ""


# ------------------------------------------------------------------ manager: basics


@pytest.fixture
def manager() -> ModelManager:
    provider = _FakeProvider([_tag("qwen:9b"), _tag("small:4b", params="4B", size_gb=2.4)])
    return ModelManager(provider, _machine(12000))


async def test_load_unload_and_swap_reach_the_provider(manager: ModelManager) -> None:
    await manager.refresh_installed()

    await manager.load("qwen:9b", ModelRole.REASONING)
    assert manager.loaded()[0]["name"] == "qwen:9b"

    await manager.swap("qwen:9b", "small:4b", ModelRole.RESUME)
    provider = manager.provider
    assert isinstance(provider, _FakeProvider)
    assert provider.unloaded == ["qwen:9b"]
    assert provider.loaded == ["qwen:9b", "small:4b"]
    assert [entry["name"] for entry in manager.loaded()] == ["small:4b"]


async def test_loading_a_model_that_is_not_installed_fails_clearly(
    manager: ModelManager,
) -> None:
    await manager.refresh_installed()

    with pytest.raises(ModelUnavailable, match="not installed"):
        await manager.load("nothing:70b", ModelRole.REASONING)


async def test_an_unreachable_backend_reports_no_models_instead_of_crashing() -> None:
    class _Dead(_FakeProvider):
        async def list_model_details(self) -> list[dict[str, Any]]:
            raise ConnectionError("connection refused")

    manager = ModelManager(_Dead(), _machine(12000))

    assert await manager.refresh_installed() == []


async def test_validation_rejects_a_configuration_naming_a_missing_model(
    manager: ModelManager,
) -> None:
    await manager.refresh_installed()
    config = AIConfiguration(roles={"reasoning": RoleConfig(model="ghost:70b")})

    results = manager.validate_configuration(config)

    assert results["reasoning"].verdict == VERDICT_BLOCK
    assert "not installed" in results["reasoning"].reasons[0]


async def test_validation_rejects_an_empty_role(manager: ModelManager) -> None:
    await manager.refresh_installed()

    results = manager.validate_configuration(AIConfiguration(roles={"reasoning": RoleConfig()}))

    assert results["reasoning"].verdict == VERDICT_BLOCK


# ------------------------------------------------------------- manager: persistence


async def test_configuration_survives_a_restart() -> None:
    store: dict[str, Any] = {}
    provider = _FakeProvider([_tag("qwen:9b")])
    manager = ModelManager(
        provider,
        _machine(12000),
        load_config=lambda: store,
        save_config=store.update,
    )
    await manager.refresh_installed()

    manager.save_configuration(
        AIConfiguration(profile="quality", roles={"reasoning": RoleConfig(model="qwen:9b")})
    )
    restored = manager.load_configuration()

    assert restored.profile == "quality"
    assert restored.roles["reasoning"].model == "qwen:9b"


async def test_with_nothing_saved_the_machine_decides() -> None:
    provider = _FakeProvider([_tag("qwen:9b")])
    manager = ModelManager(provider, _machine(6000), load_config=dict)
    await manager.refresh_installed()

    config = manager.load_configuration()

    assert config.low_vram_mode is True


# ----------------------------------------------------------- manager: OOM recovery


def test_out_of_memory_is_recognised_across_phrasings() -> None:
    assert is_out_of_memory(RuntimeError("CUDA error: out of memory"))
    assert is_out_of_memory(RuntimeError("failed to allocate VRAM"))
    assert not is_out_of_memory(ValueError("invalid json"))


async def test_an_out_of_memory_failure_retries_with_a_smaller_context(
    manager: ModelManager,
) -> None:
    await manager.refresh_installed()
    config = AIConfiguration(
        roles={"reasoning": RoleConfig(model="qwen:9b", context_tokens=16384)}
    )
    seen: list[int] = []

    async def operation(role_config: RoleConfig) -> str:
        seen.append(role_config.context_tokens)
        if len(seen) == 1:
            raise RuntimeError("CUDA out of memory")
        return "done"

    assert await manager.run_with_recovery(operation, ModelRole.REASONING, config) == "done"
    assert seen == [16384, 8192]
    assert manager.metrics["oom_count"] == 1


async def test_the_ladder_ends_on_the_fallback_model(manager: ModelManager) -> None:
    await manager.refresh_installed()
    config = AIConfiguration(
        roles={
            "reasoning": RoleConfig(
                model="qwen:9b", context_tokens=8192, fallback_model="small:4b"
            )
        }
    )
    used: list[str] = []

    async def operation(role_config: RoleConfig) -> str:
        used.append(role_config.model)
        if role_config.model != "small:4b":
            raise RuntimeError("cuda error: out of memory")
        return "done"

    assert await manager.run_with_recovery(operation, ModelRole.REASONING, config) == "done"
    assert used == ["qwen:9b", "qwen:9b", "small:4b"]
    assert manager.metrics["fallback_count"] >= 1


async def test_when_every_rung_fails_the_user_is_told_rather_than_left_guessing(
    manager: ModelManager,
) -> None:
    await manager.refresh_installed()
    config = AIConfiguration(roles={"reasoning": RoleConfig(model="qwen:9b")})

    async def always_oom(_role_config: RoleConfig) -> str:
        raise RuntimeError("out of memory")

    with pytest.raises(ModelUnavailable, match="lighter configuration"):
        await manager.run_with_recovery(always_oom, ModelRole.REASONING, config)


async def test_an_error_that_is_not_memory_related_is_not_retried(
    manager: ModelManager,
) -> None:
    """Retrying a bad prompt three times wastes a minute and fixes nothing."""
    await manager.refresh_installed()
    config = AIConfiguration(roles={"reasoning": RoleConfig(model="qwen:9b")})
    attempts = 0

    async def broken(_role_config: RoleConfig) -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("schema mismatch")

    with pytest.raises(ValueError, match="schema mismatch"):
        await manager.run_with_recovery(broken, ModelRole.REASONING, config)
    assert attempts == 1


# ---------------------------------------------------------------- manager: testing


async def test_testing_a_model_reports_measured_numbers(manager: ModelManager) -> None:
    await manager.refresh_installed()

    result = await manager.test_model("qwen:9b")

    assert result["ok"] is True
    assert result["quantization"] == "Q4_K_M"
    assert result["resident_mb"] > 0
    assert "estimated_mb" in result


async def test_a_failing_model_test_says_whether_it_was_memory(manager: ModelManager) -> None:
    await manager.refresh_installed()
    provider = manager.provider
    assert isinstance(provider, _FakeProvider)
    provider.fail_with = RuntimeError("CUDA out of memory")

    result = await manager.test_model("qwen:9b")

    assert result["ok"] is False
    assert result["out_of_memory"] is True

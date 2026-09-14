"""Ollama provider.

Uses Ollama's native ``/api/chat`` because it exposes two things the
OpenAI-compatible shim does not: JSON-Schema-constrained decoding via ``format``,
and explicit residency control via ``keep_alive``, which is what makes model
swapping on a 12 GB card workable.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from cvnator.ai.base import (
    ChatMessage,
    CompletionResult,
    LLMProvider,
    ProviderError,
)
from cvnator.observability import get_logger

log = get_logger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, timeout_s: float = 240.0, max_retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
            )
        return self._client

    async def is_available(self) -> bool:
        try:
            response = await self._http().get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        response = await self._http().get("/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
        think: bool | None = None,
    ) -> CompletionResult:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": options,
            # Residency is managed explicitly by the ResourceManager, so keep the
            # model warm here and let the manager decide when to evict it.
            "keep_alive": "10m",
        }
        if json_schema is not None:
            payload["format"] = json_schema
        if think is not None:
            # Reasoning models (qwen3.x, deepseek-r1, ...) put chain-of-thought in a
            # separate `thinking` field and can spend the entire token budget there,
            # returning empty `content`. For schema-constrained calls the structure
            # is the reasoning scaffold, so thinking is turned off explicitly.
            payload["think"] = think

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = await self._http().post("/api/chat", json=payload)
                if response.status_code == 404:
                    raise ProviderError(
                        f"Ollama does not have model {model!r}. Run: ollama pull {model}"
                    )
                response.raise_for_status()
                body = response.json()
                message = body.get("message", {})
                content = message.get("content", "") or ""
                if not content.strip() and message.get("thinking"):
                    raise ProviderError(
                        f"{model} spent its entire token budget on internal reasoning and "
                        "returned no content. Disable thinking for this call or raise "
                        "max_tokens."
                    )
                return CompletionResult(
                    text=content,
                    model=model,
                    prompt_tokens=body.get("prompt_eval_count", 0),
                    completion_tokens=body.get("eval_count", 0),
                    duration_s=time.perf_counter() - started,
                    raw=body,
                )
            except ProviderError:
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning(
                    "Ollama chat attempt %s/%s failed for %s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    model,
                    exc,
                )
        raise ProviderError(f"Ollama request failed after retries: {last_error}")

    async def load_model(self, model: str) -> None:
        """Ask Ollama to resident-load a model without generating anything."""
        try:
            await self._http().post(
                "/api/chat", json={"model": model, "messages": [], "keep_alive": "10m"}
            )
        except httpx.HTTPError as exc:  # pragma: no cover - warm-up is best effort
            log.warning("Could not preload %s: %s", model, exc)

    async def unload_model(self, model: str) -> None:
        """``keep_alive: 0`` evicts the model immediately, freeing VRAM."""
        try:
            await self._http().post(
                "/api/chat", json={"model": model, "messages": [], "keep_alive": 0}
            )
            log.info("Unloaded model %s from VRAM", model)
        except httpx.HTTPError as exc:  # pragma: no cover
            log.warning("Could not unload %s: %s", model, exc)

    async def loaded_models(self) -> dict[str, int]:
        try:
            response = await self._http().get("/api/ps", timeout=10.0)
            response.raise_for_status()
            return {
                m["name"]: int(m.get("size_vram", m.get("size", 0)))
                for m in response.json().get("models", [])
            }
        except httpx.HTTPError:
            return {}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

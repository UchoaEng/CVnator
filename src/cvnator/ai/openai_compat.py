"""OpenAI-compatible provider (llama.cpp server, vLLM, LM Studio, or a remote API).

Privacy note: this is the only path by which candidate data can leave the machine.
It refuses to run unless a base URL is explicitly configured, and it logs, once per
process, exactly which host will receive the data.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from cvnator.ai.base import ChatMessage, CompletionResult, LLMProvider, ProviderError
from cvnator.observability import get_logger

log = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_s: float = 240.0,
        max_retries: int = 2,
    ) -> None:
        if not base_url:
            raise ProviderError(
                "openai_compatible provider requires CVNATOR_OPENAI_BASE_URL to be set"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
        self._announced = False

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
            )
        return self._client

    def _announce_egress(self) -> None:
        if not self._announced:
            host = httpx.URL(self.base_url).host
            log.warning(
                "Candidate data will be sent to the remote endpoint %s. "
                "Set CVNATOR_OPENAI_BASE_URL empty to stay fully local.",
                host,
            )
            self._announced = True

    async def is_available(self) -> bool:
        try:
            response = await self._http().get("/models", timeout=5.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

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
        # `think` has no portable OpenAI-API equivalent; reasoning models on that
        # API manage their own budget, so it is accepted and ignored.
        self._announce_egress()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": False},
            }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = await self._http().post("/chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                return CompletionResult(
                    text=body["choices"][0]["message"].get("content") or "",
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    duration_s=time.perf_counter() - started,
                    raw=body,
                )
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_error = exc
                log.warning("OpenAI-compatible attempt %s failed: %s", attempt + 1, exc)
        raise ProviderError(f"OpenAI-compatible request failed after retries: {last_error}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

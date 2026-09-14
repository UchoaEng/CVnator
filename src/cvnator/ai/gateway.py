"""The single entry point the rest of the application uses to talk to models.

Callers ask for a *role* and hand over a Pydantic model describing the answer they
need. The gateway picks the provider, makes the model resident, constrains decoding
to the schema, parses and validates the result, and retries once with the validation
error fed back to the model before giving up.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from cvnator.ai.base import (
    ChatMessage,
    CompletionResult,
    LLMProvider,
    ModelRole,
    ProviderError,
    StructuredOutputError,
    extract_json,
    strictify_schema,
)
from cvnator.ai.ollama_provider import OllamaProvider
from cvnator.ai.openai_compat import OpenAICompatibleProvider
from cvnator.ai.resource_manager import ResourceManager
from cvnator.config import Settings, get_settings
from cvnator.observability import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class AIGateway:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.resources = ResourceManager(
            budget_mb=self.settings.vram_budget_mb,
            idle_unload_s=self.settings.model_idle_unload_s,
        )
        self._providers: dict[str, LLMProvider] = {}
        self._role_config: dict[ModelRole, tuple[str, str]] = {
            ModelRole.REASONING: (
                self.settings.reasoning_provider,
                self.settings.reasoning_model,
            ),
            ModelRole.VISION: (self.settings.vision_provider, self.settings.vision_model),
            ModelRole.RESUME: (self.settings.resume_provider, self.settings.resume_model),
            ModelRole.TRANSLATION: (
                self.settings.translation_provider,
                self.settings.translation_model,
            ),
        }

    # ------------------------------------------------------------------ providers

    def provider(self, name: str) -> LLMProvider:
        if name not in self._providers:
            if name == "ollama":
                self._providers[name] = OllamaProvider(
                    self.settings.ollama_base_url,
                    timeout_s=self.settings.llm_timeout_s,
                    max_retries=self.settings.llm_max_retries,
                )
            elif name in {"openai", "openai_compatible"}:
                self._providers[name] = OpenAICompatibleProvider(
                    self.settings.openai_base_url,
                    api_key=self.settings.openai_api_key,
                    timeout_s=self.settings.llm_timeout_s,
                    max_retries=self.settings.llm_max_retries,
                )
            else:
                raise ProviderError(f"Unknown AI provider: {name!r}")
        return self._providers[name]

    def model_for(self, role: ModelRole) -> tuple[LLMProvider, str]:
        provider_name, model_name = self._role_config[role]
        return self.provider(provider_name), model_name

    async def health(self) -> dict[str, Any]:
        report: dict[str, Any] = {"roles": {}, "resources": self.resources.snapshot()}
        for role, (provider_name, model_name) in self._role_config.items():
            try:
                available = await self.provider(provider_name).is_available()
            except ProviderError as exc:
                report["roles"][role.value] = {
                    "provider": provider_name,
                    "model": model_name,
                    "available": False,
                    "error": str(exc),
                }
                continue
            report["roles"][role.value] = {
                "provider": provider_name,
                "model": model_name,
                "available": available,
            }
        return report

    # ------------------------------------------------------------------ generation

    async def complete(
        self,
        role: ModelRole,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> CompletionResult:
        provider, model = self.model_for(role)
        await self.resources.acquire(provider, model, role)
        return await provider.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            think=think,
        )

    async def structured(
        self,
        role: ModelRole,
        schema_model: type[T],
        messages: list[ChatMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        repair_attempts: int = 1,
        think: bool = False,
    ) -> T:
        """Generate an instance of ``schema_model``, repairing malformed output once.

        Two non-obvious defaults, both learned the hard way against a local 9B model:

        * ``think=False``. Reasoning models route chain-of-thought into a separate
          channel and routinely exhaust the token budget there, returning empty
          content. For a schema-constrained call the schema *is* the scaffold.
        * The schema is strictified, so every field is required. Otherwise the
          constrained decoder is free to emit ``{}`` and be formally valid.
        """
        schema = strictify_schema(schema_model.model_json_schema())
        convo = list(messages)

        last_error: Exception | None = None
        for attempt in range(repair_attempts + 1):
            result = await self.complete(
                role,
                convo,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=schema,
                think=think,
            )
            try:
                payload = extract_json(result.text)
                return schema_model.model_validate(payload)
            except (StructuredOutputError, ValidationError) as exc:
                last_error = exc
                if attempt >= repair_attempts:
                    break
                log.warning(
                    "Structured output for %s failed validation (attempt %s); repairing",
                    schema_model.__name__,
                    attempt + 1,
                )
                convo = [
                    *messages,
                    ChatMessage(role="assistant", content=result.text[:4000]),
                    ChatMessage(
                        role="user",
                        content=(
                            "That response did not match the required JSON schema.\n"
                            f"Validation error:\n{exc}\n\n"
                            "Reply with ONLY a valid JSON object matching this schema:\n"
                            f"{json.dumps(schema)}"
                        ),
                    ),
                ]
        raise StructuredOutputError(
            f"Could not obtain valid {schema_model.__name__} from model: {last_error}"
        )

    async def aclose(self) -> None:
        await self.resources.release_all()
        for provider in self._providers.values():
            await provider.aclose()
        self._providers.clear()


_gateway: AIGateway | None = None


def get_gateway() -> AIGateway:
    global _gateway
    if _gateway is None:
        _gateway = AIGateway()
    return _gateway

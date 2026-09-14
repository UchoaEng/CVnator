"""Provider abstractions.

The rest of the codebase talks to *roles* (reasoning, vision, resume, translation),
never to a concrete model. Swapping Ollama for llama.cpp, vLLM or a remote
OpenAI-compatible endpoint means adding one class here, not touching the agent.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelRole(str, Enum):
    REASONING = "reasoning"
    VISION = "vision"
    RESUME = "resume"
    TRANSLATION = "translation"
    EMBEDDING = "embedding"


@dataclass(slots=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str
    images: list[str] = field(default_factory=list)  # base64-encoded, vision only

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            payload["images"] = self.images
        return payload


@dataclass(slots=True)
class CompletionResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised when a provider cannot serve a request (network, model missing, timeout)."""


class StructuredOutputError(ProviderError):
    """The model answered, but not with data matching the requested schema."""


class LLMProvider(ABC):
    """Text-in/text-out plus schema-constrained structured generation."""

    name: str = "abstract"

    @abstractmethod
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
    ) -> CompletionResult: ...

    @abstractmethod
    async def is_available(self) -> bool: ...

    async def load_model(self, model: str) -> None:  # pragma: no cover - optional capability
        """Warm a model into memory. Providers without the concept may no-op."""

    async def unload_model(self, model: str) -> None:  # pragma: no cover
        """Evict a model from VRAM. Providers without the concept may no-op."""

    async def loaded_models(self) -> dict[str, int]:  # pragma: no cover
        """Map of currently resident model name -> approximate VRAM bytes."""
        return {}

    async def aclose(self) -> None:  # pragma: no cover
        return None


def strictify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Mark every property of every object in a JSON Schema as required.

    Why this is necessary: pydantic omits fields that have defaults from
    ``required``, and a grammar-constrained decoder reads that as permission to stop
    early. A local model handed such a schema will cheerfully answer ``{}`` and be
    technically correct. Forcing every field to be present makes the model actually
    populate the structure; the Python side keeps its defaults, so nothing is lost if
    a field comes back empty.
    """
    import copy

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    strict = copy.deepcopy(schema)
    walk(strict)
    return strict


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Handles the three ways local models get this wrong: wrapping the object in a
    fenced code block, emitting a ``<think>`` preamble, and adding prose around an
    otherwise valid object.
    """
    cleaned = _THINK_BLOCK.sub("", text).strip()
    if not cleaned:
        raise StructuredOutputError("Model returned an empty response")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fenced = _JSON_BLOCK.search(cleaned)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise StructuredOutputError(f"No JSON object found in response: {cleaned[:300]!r}")

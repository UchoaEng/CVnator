"""Vision fallback: interpret a page from a screenshot when the DOM is not enough.

Used sparingly and on purpose. A screenshot costs a VLM load (which may evict the
reasoning model from VRAM), several seconds, and a large number of tokens. It earns
that cost only when the cheaper layers have genuinely failed - canvas-rendered UIs,
shadow DOM without accessible names, or a page the agent has stared at without
progress.

The VLM's job is narrow: describe the interface in structured terms. It does not
decide what to do. That decision stays with the reasoning model, which receives the
description alongside the DOM inventory.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cvnator.ai import ChatMessage, ModelRole
from cvnator.ai.gateway import AIGateway
from cvnator.observability import get_logger

log = get_logger(__name__)


class VisualElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="button, text field, dropdown, checkbox, link, heading, ...")
    label: str = Field(description="The text visible on or next to the element")
    position: str = Field(default="", description="Rough location, e.g. 'bottom right'")
    state: str = Field(default="", description="disabled, checked, filled, showing an error, ...")


class VisualReading(BaseModel):
    """What the vision model saw. Description only - never an instruction to act."""

    model_config = ConfigDict(extra="forbid")

    page_purpose: str = Field(default="", description="What this screen appears to be for")
    elements: list[VisualElement] = Field(default_factory=list)
    error_messages: list[str] = Field(default_factory=list)
    blocking_overlay: str = Field(
        default="",
        description="Any modal, cookie banner or verification challenge covering the page",
    )
    likely_next_control: str = Field(
        default="", description="The control that appears to advance the flow"
    )

    def as_prompt_block(self) -> str:
        lines = [f"VISUAL READING: {self.page_purpose}"]
        if self.blocking_overlay:
            lines.append(f"OVERLAY PRESENT: {self.blocking_overlay}")
        if self.error_messages:
            lines += [f"VISIBLE ERROR: {e}" for e in self.error_messages]
        for element in self.elements[:40]:
            bits = [f"  - {element.kind}: \"{element.label}\""]
            if element.position:
                bits.append(f"({element.position})")
            if element.state:
                bits.append(f"[{element.state}]")
            lines.append(" ".join(bits))
        if self.likely_next_control:
            lines.append(f"CONTROL THAT LIKELY ADVANCES: {self.likely_next_control}")
        return "\n".join(lines)


_VISION_SYSTEM = """\
You describe what is visible in a screenshot of a web page. You are a describer, not
an actor and not a guesser.

- List only elements you can actually see. Do not infer hidden fields.
- Copy label text exactly as rendered, including its language.
- Report any modal, cookie banner, CAPTCHA or verification challenge as
  `blocking_overlay`.
- Report visible validation or error text in `error_messages`.
- `likely_next_control` is the button or link that appears to move the flow forward.
Return only JSON matching the schema.
"""


async def read_screenshot(
    gateway: AIGateway, screenshot_path: Path, *, question: str = ""
) -> VisualReading:
    """Send one screenshot to the vision model and get a structured description back."""
    raw = await asyncio.to_thread(Path(screenshot_path).read_bytes)
    image_b64 = base64.b64encode(raw).decode("ascii")
    prompt = question or (
        "Describe this page: what it is for, every visible interactive element, any "
        "error message, and any overlay blocking interaction."
    )
    messages = [
        ChatMessage(role="system", content=_VISION_SYSTEM),
        ChatMessage(role="user", content=prompt, images=[image_b64]),
    ]
    log.info("Vision fallback engaged on %s", screenshot_path.name)
    return await gateway.structured(
        ModelRole.VISION, VisualReading, messages, temperature=0.0, max_tokens=2048
    )

"""Page perception: turning a live page into something a language model can reason over.

The strategy is layered, cheapest first, exactly as the architecture requires:

    1. DOM  ->  2. Accessibility tree  ->  3. Browser state  ->  4. Screenshot  ->  5. VLM

Layers 1-3 are free and deterministic and answer most questions. The screenshot and
the vision model are a fallback, reached only when the cheaper layers leave the agent
unable to identify what it needs - sending an image on every step would burn VRAM,
tokens and wall-clock for no gain.

What comes out is *evidence*, not instructions. The perception layer never decides
what to click; it produces a labelled inventory and hands it to the reasoning model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page

from cvnator.observability import get_logger

log = get_logger(__name__)

MAX_ELEMENTS = 120
MAX_TEXT_CHARS = 6000

#: JavaScript that walks the DOM once and returns a labelled inventory of everything
#: interactive plus the visible text. Doing it in one evaluate call keeps the page
#: from mutating between reads.
_INVENTORY_JS = r"""
(maxElements) => {
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') {
      return false;
    }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };

  const labelFor = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim();
    const labelledBy = el.getAttribute('aria-labelledby');
    if (labelledBy) {
      const parts = labelledBy.split(/\s+/)
        .map((id) => document.getElementById(id))
        .filter(Boolean)
        .map((n) => n.innerText.trim());
      if (parts.length) return parts.join(' ');
    }
    if (el.id) {
      const explicit = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (explicit) return explicit.innerText.trim();
    }
    const wrapping = el.closest('label');
    if (wrapping) return wrapping.innerText.trim();
    // Fall back to the nearest preceding label or legend: this is how a human reads a
    // form whose markup never associated the two.
    let node = el.previousElementSibling;
    let hops = 0;
    while (node && hops < 4) {
      if (/^(LABEL|LEGEND|SPAN|P|DIV|H[1-6])$/.test(node.tagName)) {
        const text = (node.innerText || '').trim();
        if (text && text.length < 160) return text;
      }
      node = node.previousElementSibling;
      hops += 1;
    }
    const fieldset = el.closest('fieldset');
    if (fieldset) {
      const legend = fieldset.querySelector('legend');
      if (legend) return legend.innerText.trim();
    }
    return el.getAttribute('placeholder') || el.getAttribute('name') || '';
  };

  // A selector is only useful if it resolves to exactly one element. Forms routinely
  // share a `name` between controls - two submit buttons called "action", a radio
  // group - so every candidate is tested for uniqueness before it is accepted, and
  // a full structural path is used when it is not unique.
  const structuralPath = (el) => {
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && node.tagName !== 'HTML') {
      let part = node.tagName.toLowerCase();
      if (node.parentElement) {
        const siblings = Array.from(node.parentElement.children)
          .filter((c) => c.tagName === node.tagName);
        if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
      }
      parts.unshift(part);
      node = node.parentElement;
    }
    return parts.join(' > ');
  };

  const isUniqueFor = (selector, el) => {
    try {
      const matches = document.querySelectorAll(selector);
      return matches.length === 1 && matches[0] === el;
    } catch {
      return false;
    }
  };

  const selectorFor = (el) => {
    const tag = el.tagName.toLowerCase();
    const candidates = [];
    if (el.id) candidates.push(`#${CSS.escape(el.id)}`);
    const testId = el.getAttribute('data-testid');
    if (testId) candidates.push(`[data-testid="${CSS.escape(testId)}"]`);
    const name = el.getAttribute('name');
    if (name) {
      const value = el.getAttribute('value');
      // Disambiguates submit buttons and radio groups that share a name.
      if (value) {
        candidates.push(
          `${tag}[name="${CSS.escape(name)}"][value="${CSS.escape(value)}"]`
        );
      }
      candidates.push(`${tag}[name="${CSS.escape(name)}"]`);
    }
    for (const candidate of candidates) {
      if (isUniqueFor(candidate, el)) return candidate;
    }
    return structuralPath(el);
  };

  const SELECTOR = [
    'input:not([type=hidden])', 'textarea', 'select', 'button',
    'a[href]', '[role=button]', '[role=link]', '[role=checkbox]', '[role=radio]',
    '[role=combobox]', '[role=tab]', '[contenteditable=true]',
  ].join(',');

  const elements = [];
  let index = 0;
  for (const el of document.querySelectorAll(SELECTOR)) {
    if (elements.length >= maxElements) break;
    if (!isVisible(el)) continue;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    const label = labelFor(el);

    // fieldset > legend + per-option <label> is the standard accessible pattern for a
    // radio group. labelFor stops at the wrapping label and returns "Yes"/"No", so the
    // legend - which carries the actual question AND its required marker - is invisible.
    // Capture it separately for radios and checkboxes.
    let groupLabel = '';
    if (el.type === 'radio' || el.type === 'checkbox') {
      const fieldset = el.closest('fieldset');
      const legend = fieldset ? fieldset.querySelector('legend') : null;
      if (legend) groupLabel = (legend.innerText || '').trim();
    }

    const marksRequired = (text) => !!text && (
      /[*∗]\s*$/.test(text.trim())
      || /\(\s*(required|obrigat[óo]rio|requerido)\s*\)/i.test(text)
    );
    const starRequired = marksRequired(label) || marksRequired(groupLabel);
    const entry = {
      element_id: 'e' + (index++),
      tag,
      type: type || null,
      role: el.getAttribute('role') || null,
      label: label.slice(0, 200),
      group_label: groupLabel.slice(0, 200) || null,
      text: (el.innerText || el.value || '').trim().slice(0, 200),
      name: el.getAttribute('name') || null,
      placeholder: el.getAttribute('placeholder') || null,
      required: el.required === true
                || el.getAttribute('aria-required') === 'true'
                || starRequired,
      required_source: el.required === true ? 'attribute'
                       : (starRequired ? 'label' : null),
      disabled: el.disabled === true,
      value: (el.type === 'password') ? null
             : (typeof el.value === 'string' ? el.value.slice(0, 200) : null),
      // The true length, since `value` above is clipped for transport. Reporting the
      // clipped length would tell the agent its own answer had been truncated.
      value_length: (el.type === 'password' || typeof el.value !== 'string')
                    ? null : el.value.length,
      checked: (el.type === 'checkbox' || el.type === 'radio') ? el.checked : null,
      href: el.getAttribute('href'),
      selector: selectorFor(el),
      // HTML5 constraint validation. A browser refuses to submit an invalid form and
      // reports why in a native bubble that never appears in the DOM - so without
      // this the agent sees a click that "did nothing" and no reason for it.
      invalid: (typeof el.checkValidity === 'function') ? !el.checkValidity() : false,
      validation_message: el.validationMessage || null,
      constraints: (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') ? {
        min: el.getAttribute('min'),
        max: el.getAttribute('max'),
        step: el.getAttribute('step'),
        pattern: el.getAttribute('pattern'),
        minlength: el.getAttribute('minlength'),
        maxlength: el.getAttribute('maxlength'),
      } : null,
    };
    if (tag === 'select') {
      entry.options = Array.from(el.options).slice(0, 40)
        .map((o) => ({ value: o.value, label: o.text.trim() }));
    }
    elements.push(entry);
  }

  // Validation and status messages, which is how a page reports a rejected step.
  const alerts = Array.from(
    document.querySelectorAll('[role=alert], .error, .err, .invalid-feedback, [aria-invalid=true]')
  ).filter(isVisible).map((el) => (el.innerText || '').trim()).filter(Boolean).slice(0, 15);

  // Native constraint-validation failures are surfaced as alerts too, so the agent
  // reads them the same way it reads a server-rendered error.
  for (const entry of elements) {
    if (entry.invalid && entry.validation_message) {
      alerts.push(`[${entry.element_id}] ${entry.label || entry.name || 'field'}: `
                  + entry.validation_message);
    }
  }

  return {
    url: location.href,
    title: document.title,
    text: (document.body ? document.body.innerText : '').slice(0, 20000),
    elements,
    alerts,
    forms: document.forms.length,
    lang: document.documentElement.lang || '',
  };
}
"""


@dataclass(slots=True)
class PageElement:
    element_id: str
    tag: str
    type: str | None
    role: str | None
    label: str
    text: str
    name: str | None
    placeholder: str | None
    required: bool
    disabled: bool
    value: str | None
    checked: bool | None
    href: str | None
    selector: str
    options: list[dict[str, str]] = field(default_factory=list)
    invalid: bool = False
    validation_message: str | None = None
    constraints: dict[str, Any] | None = None
    required_source: str | None = None
    value_length: int | None = None
    #: For a radio or checkbox inside a fieldset, the legend's question text.
    group_label: str | None = None

    def describe(self) -> str:
        """One compact line per element, which is what the reasoning model sees."""
        kind = self.role or (f"{self.tag}:{self.type}" if self.type else self.tag)
        bits = [f"[{self.element_id}] {kind}"]
        if self.group_label and self.group_label != self.label:
            bits.append(f'question="{self.group_label}"')
        if self.label:
            bits.append(f'label="{self.label}"')
        elif self.text:
            bits.append(f'text="{self.text[:80]}"')
        if self.required:
            bits.append("REQUIRED")
        if self.value:
            # Say so when the *display* is truncated. Showing a clipped value with no
            # marker reads as "the field content is truncated", and the agent rewrites
            # a field that was perfectly fine.
            full_length = self.value_length if self.value_length is not None else len(self.value)
            if full_length > 60:
                bits.append(
                    f'current="{self.value[:60]}..." '
                    f"({full_length} chars total, shown clipped)"
                )
            else:
                bits.append(f'current="{self.value}"')
        if self.checked is not None:
            bits.append(f"checked={self.checked}")
        if self.disabled:
            bits.append("DISABLED")
        if self.options:
            labels = ", ".join(o["label"] for o in self.options[:12] if o["label"])
            bits.append(f"options=[{labels}]")
        if self.constraints:
            rules = ", ".join(
                f"{k}={v}" for k, v in self.constraints.items() if v not in (None, "")
            )
            if rules:
                bits.append(f"constraints({rules})")
        if self.invalid and self.validation_message:
            bits.append(f'INVALID: "{self.validation_message}"')
        return " ".join(bits)


@dataclass(slots=True)
class PageState:
    url: str
    title: str
    text: str
    elements: list[PageElement]
    alerts: list[str]
    lang: str
    form_count: int
    fingerprint: str

    def element_by_id(self, element_id: str) -> PageElement | None:
        return next((e for e in self.elements if e.element_id == element_id), None)

    def interactive_summary(self, limit: int = MAX_ELEMENTS) -> str:
        return "\n".join(e.describe() for e in self.elements[:limit])

    def visible_text(self, limit: int = MAX_TEXT_CHARS) -> str:
        collapsed = re.sub(r"\n{3,}", "\n\n", self.text)
        return collapsed[:limit]

    def has_password_field(self) -> bool:
        return any(e.type == "password" for e in self.elements)

    def form_progress(self) -> str:
        """A checklist of required fields and whether each is already satisfied.

        The element inventory technically contains this, but a model reading 20 lines
        of attributes reliably loses track and refills fields it has already filled.
        Stating the remaining work explicitly is far cheaper than the wasted cycles.
        """
        required = [e for e in self.elements if e.required and not e.disabled]
        if not required:
            return ""

        # A radio group is a single requirement satisfied by any one of its options.
        # Counting each option separately reports a chosen group as half-finished.
        radio_groups: dict[str, list[PageElement]] = {}
        singles: list[PageElement] = []
        for element in required:
            if element.type == "radio" and element.name:
                radio_groups.setdefault(element.name, []).append(element)
            else:
                singles.append(element)

        done: list[str] = []
        todo: list[str] = []
        for element in singles:
            name = element.label or element.name or element.element_id
            if element.type == "checkbox":
                (done if element.checked else todo).append(f"[{element.element_id}] {name}")
            elif (element.value or "").strip():
                done.append(f"[{element.element_id}] {name}")
            else:
                todo.append(f"[{element.element_id}] {name}")

        for group_name, options in radio_groups.items():
            label = options[0].group_label or options[0].label or group_name
            chosen = next((o for o in options if o.checked), None)
            ids = "/".join(o.element_id for o in options)
            if chosen is not None:
                done.append(f"[{chosen.element_id}] {label}")
            else:
                todo.append(f"[{ids}] {label} (choose one)")

        required_count = len(singles) + len(radio_groups)
        lines = [f"REQUIRED FIELDS: {len(done)} of {required_count} done"]
        if todo:
            lines.append("  STILL EMPTY: " + "; ".join(todo[:12]))
        if done:
            lines.append("  ALREADY FILLED (do not refill): " + "; ".join(done[:12]))
        return "\n".join(lines)

    def as_prompt_block(self) -> str:
        parts = [
            f"URL: {self.url}",
            f"TITLE: {self.title}",
            f"PAGE LANGUAGE: {self.lang or 'unknown'}",
        ]
        progress = self.form_progress()
        if progress:
            parts.append(progress)
        if self.alerts:
            parts.append("VALIDATION MESSAGES ON PAGE:")
            parts += [f"  ! {a}" for a in self.alerts]
        parts += [
            "",
            "INTERACTIVE ELEMENTS:",
            self.interactive_summary(),
            "",
            "VISIBLE TEXT (truncated):",
            self.visible_text(2500),
        ]
        return "\n".join(parts)


def compute_fingerprint(url: str, title: str, elements: list[dict[str, Any]], text: str) -> str:
    """A stable identity for "the page as it currently is".

    Used by the watchdog to tell "nothing happened" from "the page moved on", so it
    intentionally excludes volatile values (typed input, timestamps in body text) and
    keys on structure instead.
    """
    skeleton = [
        f"{e.get('tag')}|{e.get('type')}|{e.get('role')}|{(e.get('label') or '')[:60]}"
        for e in elements
    ]
    normalized_text = re.sub(r"\d", "#", re.sub(r"\s+", " ", text or ""))[:3000]
    payload = json.dumps(
        {
            "url": url.split("?")[0],
            "title": title,
            "skeleton": skeleton,
            "text": normalized_text,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


async def observe(page: Page, *, max_elements: int = MAX_ELEMENTS) -> PageState:
    """Layers 1-3: DOM inventory, derived accessibility labels, and browser state."""
    raw: dict[str, Any] = await page.evaluate(_INVENTORY_JS, max_elements)
    elements = [
        PageElement(
            element_id=item["element_id"],
            tag=item["tag"],
            type=item.get("type"),
            role=item.get("role"),
            label=item.get("label") or "",
            text=item.get("text") or "",
            name=item.get("name"),
            placeholder=item.get("placeholder"),
            required=bool(item.get("required")),
            disabled=bool(item.get("disabled")),
            value=item.get("value"),
            checked=item.get("checked"),
            href=item.get("href"),
            selector=item["selector"],
            options=item.get("options") or [],
            group_label=item.get("group_label"),
            required_source=item.get("required_source"),
            value_length=item.get("value_length"),
            invalid=bool(item.get("invalid")),
            validation_message=item.get("validation_message"),
            constraints=item.get("constraints"),
        )
        for item in raw["elements"]
    ]
    return PageState(
        url=raw["url"],
        title=raw["title"],
        text=raw["text"],
        elements=elements,
        alerts=raw.get("alerts", []),
        lang=raw.get("lang", ""),
        form_count=raw.get("forms", 0),
        fingerprint=compute_fingerprint(raw["url"], raw["title"], raw["elements"], raw["text"]),
    )


async def accessibility_snapshot(page: Page) -> dict[str, Any]:
    """Layer 2 in its raw form, for when the DOM inventory is ambiguous.

    Playwright removed ``page.accessibility`` in 1.52, so this goes to Chromium
    directly over CDP and rebuilds the tree from the flat node list CDP returns.
    """
    try:
        session = await page.context.new_cdp_session(page)
        try:
            await session.send("Accessibility.enable")
            payload = await session.send("Accessibility.getFullAXTree")
        finally:
            await session.detach()
    except Exception as exc:  # pragma: no cover - browser-dependent
        log.warning("Accessibility snapshot failed: %s", exc)
        return {}

    nodes = payload.get("nodes", [])
    if not nodes:
        return {}

    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if node.get("ignored"):
            continue
        by_id[node["nodeId"]] = {
            "role": (node.get("role") or {}).get("value", ""),
            "name": (node.get("name") or {}).get("value", ""),
            "value": (node.get("value") or {}).get("value", ""),
            "_children": node.get("childIds", []),
            "children": [],
        }

    # CDP hands back a flat list plus child ids; reassemble it into a real tree.
    child_ids = {cid for node in by_id.values() for cid in node["_children"]}
    for node in by_id.values():
        node["children"] = [by_id[cid] for cid in node["_children"] if cid in by_id]
        node.pop("_children", None)

    roots = [node_id for node_id in by_id if node_id not in child_ids]
    if not roots:
        return {}
    if len(roots) == 1:
        return by_id[roots[0]]
    return {"role": "RootWebArea", "name": "", "children": [by_id[r] for r in roots]}


def flatten_accessibility(node: dict[str, Any], depth: int = 0, limit: int = 200) -> list[str]:
    """Flatten an a11y tree to indented ``role "name"`` lines."""
    if not node or limit <= 0:
        return []
    lines: list[str] = []
    role = node.get("role", "")
    name = (node.get("name") or "").strip()
    if role and (name or role not in {"generic", "none"}):
        lines.append(f"{'  ' * depth}{role} \"{name[:80]}\"")
    for child in node.get("children", []) or []:
        if len(lines) >= limit:
            break
        lines.extend(flatten_accessibility(child, depth + 1, limit - len(lines)))
    return lines


def needs_visual_fallback(state: PageState) -> tuple[bool, str]:
    """Decide whether the cheap layers have failed and a screenshot is justified.

    Kept deliberately strict. Canvas-rendered or shadow-heavy UIs genuinely need
    vision; an ordinary form that the agent simply has not read carefully does not.
    """
    if not state.elements:
        return True, "no interactive elements found in the DOM"
    unlabelled = [e for e in state.elements if not e.label and not e.text]
    if len(unlabelled) >= max(4, len(state.elements) // 2):
        return True, "more than half of the interactive elements have no discoverable label"
    if len(state.text.strip()) < 40 and len(state.elements) < 3:
        return True, "page has almost no text and almost no elements"
    return False, ""

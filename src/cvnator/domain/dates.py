"""Tolerant date parsing for CV data, which is never well formed."""

from __future__ import annotations

import re
from datetime import date

_CURRENT_TOKENS = {
    "",
    "present",
    "current",
    "atual",
    "hoje",
    "now",
    "actualidad",
    "presente",
    "today",
}

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "fev": 2,
    "mar": 3,
    "apr": 4,
    "abr": 4,
    "may": 5,
    "mai": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "ago": 8,
    "sep": 9,
    "set": 9,
    "oct": 10,
    "out": 10,
    "nov": 11,
    "dec": 12,
    "dez": 12,
}


def parse_partial_date(value: str | None) -> tuple[int, int] | None:
    """Return ``(year, month)`` from a messy date string, or ``None`` if unparseable."""
    if value is None:
        return None
    text = value.strip().lower()
    if text in _CURRENT_TOKENS:
        return None

    iso = re.match(r"^(\d{4})[-/](\d{1,2})", text)
    if iso:
        return int(iso.group(1)), min(max(int(iso.group(2)), 1), 12)

    br = re.match(r"^(\d{1,2})[-/](\d{4})$", text)
    if br:
        return int(br.group(2)), min(max(int(br.group(1)), 1), 12)

    named = re.match(r"^([a-zà-ÿ]{3,})\.?\s*[/\-, ]\s*(\d{4})$", text)
    if named:
        month = _MONTHS.get(named.group(1)[:3])
        if month:
            return int(named.group(2)), month

    year_only = re.match(r"^(\d{4})$", text)
    if year_only:
        return int(year_only.group(1)), 1
    return None


def months_between(start: str | None, end: str | None) -> int:
    """Whole months between two partial dates. ``end`` empty/None means "until today"."""
    start_parsed = parse_partial_date(start)
    if start_parsed is None:
        return 0
    end_parsed = parse_partial_date(end)
    if end_parsed is None:
        today = date.today()
        end_parsed = (today.year, today.month)
    months = (end_parsed[0] - start_parsed[0]) * 12 + (end_parsed[1] - start_parsed[1])
    return max(months, 0)

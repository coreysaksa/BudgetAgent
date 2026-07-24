"""Parse a natural-language transaction lookback window from a chat message.

The chat grounds its reply on a spending snapshot covering the last N days.
By default that window is 30 days, but the user can widen it conversationally
("looking back 60 days", "past 6 months", "over the last year"). This module
turns such phrases into a day count deterministically (no LLM round-trip) so the
snapshot call is predictable and testable.
"""
from __future__ import annotations

import re

DEFAULT_LOOKBACK_DAYS = 30
#: Upper bound so a stray "10 years" can't blow up the snapshot / prompt size.
MAX_LOOKBACK_DAYS = 730

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

#: Approximate days per unit (calendar months/years are normalised to keep the
#: window arithmetic simple and stable across runs).
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

_NUMBER = r"\d+|" + "|".join(_WORD_NUMBERS)
_UNIT = r"(?P<unit>day|week|month|year)s?"

# Strong signal: an explicit quantity immediately followed by a unit,
# e.g. "60 days", "6-month", "three weeks". Bare "a/an" is intentionally
# excluded here so phrases like "have a good day" don't collapse the window.
_QTY_UNIT = re.compile(rf"\b(?P<qty>{_NUMBER})[\s-]+{_UNIT}\b", re.IGNORECASE)

# Lookback phrasing with a possibly-omitted quantity, e.g. "last month",
# "past year", "over the last 6 months", "looking back a year". Here a bare
# unit (or "a"/"an") is unambiguous because a lookback keyword precedes it.
_KEYWORD = (
    r"(?:last|past|previous|prior|trailing|"
    r"(?:look(?:ing)?|go(?:ing)?)\s+back|back|since)"
)
_KEYWORD_UNIT = re.compile(
    rf"\b{_KEYWORD}\s+(?:the\s+)?"
    rf"(?:(?P<qty>{_NUMBER}|a|an)\s+)?{_UNIT}\b",
    re.IGNORECASE,
)


def _qty_value(qty: str | None) -> int:
    if not qty:
        return 1
    q = qty.lower()
    if q in ("a", "an"):
        return 1
    if q.isdigit():
        return int(q)
    return _WORD_NUMBERS.get(q, 1)


def _clamp(days: int) -> int:
    return max(1, min(days, MAX_LOOKBACK_DAYS))


def parse_lookback_days(
    message: str, default: int = DEFAULT_LOOKBACK_DAYS
) -> int:
    """Return the lookback window (in days) requested by ``message``.

    Falls back to ``default`` (30 days) when no lookback phrase is present.
    The result is clamped to ``[1, MAX_LOOKBACK_DAYS]``.
    """
    if not message:
        return default

    match = _QTY_UNIT.search(message) or _KEYWORD_UNIT.search(message)
    if not match:
        return default

    qty = _qty_value(match.groupdict().get("qty"))
    unit = match.group("unit").lower()
    return _clamp(qty * _UNIT_DAYS[unit])

"""Decimal helpers for deterministic financial calculations."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from typing import Any

ZERO = Decimal("0")
CENT = Decimal("0.01")
HALF_CENT = Decimal("0.005")


def decimal_value(value: Any) -> Decimal:
    """Convert JSON-compatible numeric input without inheriting binary float error."""
    if value is None or value == "":
        return ZERO
    return Decimal(str(value))


def money(value: Any) -> Decimal:
    """Convert and round a monetary value to cents."""
    return decimal_value(value).quantize(CENT, rounding=ROUND_HALF_UP)


def money_up(value: Decimal) -> Decimal:
    """Round a required payment upward so a deadline is not missed by a cent."""
    return value.quantize(CENT, rounding=ROUND_CEILING)


def json_money(value: Decimal) -> float:
    """Return a cent-rounded float at the existing JSON contract boundary."""
    return float(money(value))

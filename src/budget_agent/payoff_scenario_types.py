"""Typed intermediate results for payoff-scenario calculation stages."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass
class ScenarioCapacity:
    today: date
    spending: list[dict[str, Any]]
    utility_forecast: dict[str, Any]
    baseline_extra: float
    essential_delta: float
    spending_savings: float
    safe_before_floor: float
    safe_extra: float
    allocation_percent: float
    has_debt: bool
    portfolio_rows: list[dict[str, Any]]
    requested_extra: float
    minimum_total: float
    streams: list[dict[str, Any]]
    uses_estimated_income: bool
    extra_debt_total: float
    extra_goal_total: float
    extra_unassigned_total: float
    extra_payments: dict[str, float]
    plan: dict[str, Any] | None
    regular_income: float
    non_debt_total: float


@dataclass
class ScenarioFeasibility:
    reasons: list[str]
    status: str
    feasible: bool | None
    shortfall: float
    underwater_limit: float
    underwater_eligible: bool
    minimum_survival: float

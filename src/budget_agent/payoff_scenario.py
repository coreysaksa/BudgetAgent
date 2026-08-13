"""Deterministic what-if inputs and feasibility for credit-card payoff plans."""
from __future__ import annotations

import calendar
from datetime import date
from statistics import median
from typing import Any, TypedDict

from .payoff import payoff_from_snapshot

_VARIABLE_ESSENTIALS = {
    "groceries",
    "fuel",
    "healthcare",
    "tolls",
    "transit",
}
_DINING = {"dining", "coffee", "delivery"}
_FREQUENCIES = {"one_time", "monthly", "quarterly", "annual", "custom"}
_STABLE_UTILITIES = {"internet", "cell_phone"}
_SEASONAL_UTILITIES = {"electric", "gas_utility", "water"}
_UTILITY_SUBCATEGORIES = _STABLE_UTILITIES | _SEASONAL_UTILITIES


class UtilityHistorySnapshot(TypedDict, total=False):
    """Longer analyzer snapshot used only for deterministic utility forecasting."""

    period_days: int | float
    spending_tree: list[dict[str, Any]]


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _add_months(value: date, count: int) -> date:
    total = value.year * 12 + value.month - 1 + count
    year, month_index = divmod(total, 12)
    month = month_index + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _spending_rows(
    analysis: dict[str, Any],
    adjustments: dict[str, float],
) -> list[dict[str, Any]]:
    period_days = max(1.0, float(analysis.get("period_days") or 30.0))
    rows: list[dict[str, Any]] = []
    for bucket in analysis.get("spending_tree") or []:
        bucket_name = str(bucket.get("bucket") or "")
        for category in bucket.get("categories") or []:
            for subcategory in category.get("subcategories") or []:
                key = str(subcategory.get("subcategory") or "other")
                current = max(
                    0.0,
                    float(subcategory.get("total") or 0.0) * 30.0 / period_days,
                )
                adjustable = (
                    key not in _UTILITY_SUBCATEGORIES
                    and (
                        bucket_name == "discretionary"
                        or key in _VARIABLE_ESSENTIALS
                    )
                )
                minimum = current * 0.7 if key in _VARIABLE_ESSENTIALS else 0.0
                default = current
                if adjustable and bucket_name == "discretionary":
                    default = current * (0.75 if key in _DINING else 0.9)
                proposed = (
                    max(0.0, float(adjustments[key]))
                    if key in adjustments and adjustable
                    else default
                )
                rows.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "bucket": bucket_name,
                        "current_monthly": round(current, 2),
                        "proposed_monthly": round(proposed, 2),
                        "minimum_monthly": round(minimum, 2),
                        "adjustable": adjustable,
                    }
                )
    return rows


def _utility_forecast(
    spending: list[dict[str, Any]],
    utility_history: UtilityHistorySnapshot | None,
    *,
    start: date,
) -> dict[str, Any]:
    stable_monthly = sum(
        row["current_monthly"] for row in spending if row["key"] in _STABLE_UTILITIES
    )
    current_seasonal = sum(
        row["current_monthly"] for row in spending if row["key"] in _SEASONAL_UTILITIES
    )
    monthly_totals: dict[str, float] = {}
    history_days = max(
        1,
        int(float((utility_history or {}).get("period_days") or 730)),
    )
    history_start = date.fromordinal(max(1, start.toordinal() - history_days + 1))
    partial_start_month = (
        history_start.strftime("%Y-%m") if history_start.day > 1 else None
    )
    current_month = start.strftime("%Y-%m")
    for bucket in (utility_history or {}).get("spending_tree") or []:
        for category in bucket.get("categories") or []:
            for subcategory in category.get("subcategories") or []:
                if subcategory.get("subcategory") not in _SEASONAL_UTILITIES:
                    continue
                for transaction in subcategory.get("transactions") or []:
                    when = _parse_date(transaction.get("date"))
                    try:
                        amount = abs(float(transaction.get("amount") or 0.0))
                    except (TypeError, ValueError):
                        continue
                    if when is None or amount <= 0:
                        continue
                    month = when.strftime("%Y-%m")
                    if month == current_month or month == partial_start_month:
                        continue
                    monthly_totals[month] = monthly_totals.get(month, 0.0) + amount

    history_months = len(monthly_totals)
    if history_months >= 12:
        safety_margin = 0.10
        confidence = "high"
    elif history_months >= 6:
        safety_margin = 0.15
        confidence = "medium"
    else:
        safety_margin = 0.20
        confidence = "low"

    historical_values = list(monthly_totals.values())
    fallback = median(historical_values) if historical_values else current_seasonal
    fallback = max(current_seasonal, fallback)
    values_by_calendar_month: dict[int, list[float]] = {}
    for month, amount in monthly_totals.items():
        calendar_month = int(month[-2:])
        values_by_calendar_month.setdefault(calendar_month, []).append(amount)

    forecasts: list[dict[str, Any]] = []
    base_values: list[float] = []
    for offset in range(1, 13):
        forecast_month = _add_months(start.replace(day=1), offset)
        comparable = values_by_calendar_month.get(forecast_month.month) or []
        seasonal_amount = median(comparable) if comparable else fallback
        protected_amount = seasonal_amount * (1.0 + safety_margin)
        base_values.append(seasonal_amount)
        forecasts.append(
            {
                "month": forecast_month.strftime("%Y-%m"),
                "seasonal_forecast": round(seasonal_amount, 2),
                "protected_seasonal_reserve": round(protected_amount, 2),
                "total_utility_reserve": round(stable_monthly + protected_amount, 2),
                "basis": "same_calendar_month" if comparable else "median_fallback",
            }
        )

    level_reserve = sum(base_values) / len(base_values) if base_values else fallback
    protected_reserve = level_reserve * (1.0 + safety_margin)
    incremental_reserve = max(0.0, protected_reserve - current_seasonal)
    protected_monthly_values = [
        value * (1.0 + safety_margin) for value in base_values
    ]
    return {
        "stable_monthly_amount": round(stable_monthly, 2),
        "current_monthly_seasonal_baseline": round(current_seasonal, 2),
        "level_monthly_seasonal_reserve": round(level_reserve, 2),
        "recommended_protected_monthly_reserve": round(protected_reserve, 2),
        "recommended_total_monthly_utility_reserve": round(
            stable_monthly + protected_reserve, 2
        ),
        "incremental_monthly_reserve": round(incremental_reserve, 2),
        "low_forecast": round(
            min(protected_monthly_values) if protected_monthly_values else protected_reserve,
            2,
        ),
        "high_forecast": round(
            max(protected_monthly_values) if protected_monthly_values else protected_reserve,
            2,
        ),
        "confidence": confidence,
        "history_months": history_months,
        "safety_margin_percentage": round(safety_margin * 100.0, 2),
        "next_12_month_forecasts": forecasts,
    }


def _income_occurrences(
    streams: list[dict[str, Any]],
    start: date,
    horizon_months: int = 120,
) -> tuple[list[dict[str, Any]], dict[str, float], bool]:
    horizon_end = _add_months(start, horizon_months)
    normalized: list[dict[str, Any]] = []
    payments: dict[str, float] = {}
    uses_estimated = False
    for index, stream in enumerate(streams):
        name = str(stream.get("name") or f"Extra income {index + 1}").strip()
        amount = max(0.0, float(stream.get("amount") or 0.0))
        frequency = str(stream.get("frequency") or "one_time").lower()
        if frequency not in _FREQUENCIES:
            frequency = "one_time"
        status = str(stream.get("status") or "estimated").lower()
        if status not in {"confirmed", "estimated"}:
            status = "estimated"
        debt_percent = min(100.0, max(0.0, float(stream.get("debt_percent") or 100.0)))
        first = _parse_date(stream.get("first_date"))
        end = _parse_date(stream.get("end_date"))
        custom_dates = [
            value
            for value in (_parse_date(item) for item in stream.get("dates") or [])
            if value is not None
        ]
        dates: list[date] = []
        if frequency == "custom":
            dates = custom_dates
        elif first is not None:
            step = {"monthly": 1, "quarterly": 3, "annual": 12}.get(frequency)
            if step is None:
                dates = [first]
            else:
                cursor = first
                while cursor <= horizon_end and (end is None or cursor <= end):
                    dates.append(cursor)
                    cursor = _add_months(cursor, step)
        dates = sorted({item for item in dates if start <= item <= horizon_end})
        allocated = amount * debt_percent / 100.0
        for when in dates:
            month = when.strftime("%Y-%m")
            payments[month] = payments.get(month, 0.0) + allocated
        if status == "estimated" and dates and allocated > 0:
            uses_estimated = True
        normalized.append(
            {
                "id": str(stream.get("id") or f"extra-{index + 1}"),
                "name": name,
                "amount": round(amount, 2),
                "frequency": frequency,
                "first_date": first.isoformat() if first else None,
                "end_date": end.isoformat() if end else None,
                "dates": [item.isoformat() for item in custom_dates],
                "status": status,
                "debt_percent": round(debt_percent, 2),
                "occurrences": [item.isoformat() for item in dates[:24]],
            }
        )
    return normalized, payments, uses_estimated


def suggest_extra_income(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Suggest irregular non-payroll income streams from observed deposits."""
    suggestions: list[dict[str, Any]] = []
    excluded = ("payroll", "salary", "refund", "interest", "transfer")
    today = date.today()
    for index, source in enumerate(analysis.get("income_tree") or []):
        name = str(source.get("source") or "").strip()
        if not name or any(term in name.lower() for term in excluded):
            continue
        transactions = [
            (when, float(item.get("amount") or 0.0))
            for item in source.get("transactions") or []
            if (when := _parse_date(item.get("date")))
            and float(item.get("amount") or 0.0) > 0
        ]
        if not transactions:
            continue
        transactions.sort()
        amounts = [amount for _, amount in transactions]
        dates = [when for when, _ in transactions]
        frequency = "one_time"
        step: int | None = None
        if len(dates) >= 2:
            intervals = [
                (later - earlier).days
                for earlier, later in zip(dates, dates[1:])
            ]
            typical_gap = median(intervals) if intervals else 365
            if typical_gap <= 35:
                continue
            if typical_gap <= 45:
                frequency, step = "monthly", 1
            elif typical_gap <= 120:
                frequency, step = "quarterly", 3
            elif 250 <= typical_gap <= 450:
                frequency, step = "annual", 12
        first = dates[-1]
        while step is not None and first < today:
            first = _add_months(first, step)
        if step is None and first < today:
            first_value: str | None = None
        else:
            first_value = first.isoformat()
        suggestions.append(
            {
                "id": f"suggested-{index + 1}",
                "name": name,
                "amount": round(median(amounts), 2),
                "frequency": frequency,
                "first_date": first_value,
                "end_date": None,
                "dates": [],
                "status": "estimated",
                "debt_percent": 100.0,
            }
        )
    return suggestions


def build_payoff_scenario(
    analysis: dict[str, Any],
    cash_flow_plan: dict[str, Any],
    goals: list[dict[str, Any]],
    *,
    utility_history: UtilityHistorySnapshot | None = None,
    extra_income: list[dict[str, Any]] | None = None,
    spending_adjustments: dict[str, float] | None = None,
    debt_allocation_percent: float = 100.0,
    monthly_debt_extra: float | None = None,
) -> dict[str, Any]:
    """Build an editable proposal and a deterministic payoff feasibility result."""
    spending = _spending_rows(analysis, spending_adjustments or {})
    today = date.today()
    utility_forecast = _utility_forecast(
        spending,
        utility_history,
        start=today,
    )
    utility_reserve_increment = utility_forecast["incremental_monthly_reserve"]
    essential_delta = sum(
        row["current_monthly"] - row["proposed_monthly"]
        for row in spending
        if row["adjustable"] and row["bucket"] == "mandatory"
    )
    proposed_discretionary = sum(
        row["proposed_monthly"]
        for row in spending
        if row["adjustable"] and row["bucket"] == "discretionary"
    )
    current_adjustable = sum(
        row["current_monthly"] for row in spending if row["adjustable"]
    )
    proposed_adjustable = sum(
        row["proposed_monthly"] for row in spending if row["adjustable"]
    )
    spending_savings = current_adjustable - proposed_adjustable
    baseline_extra = max(
        0.0, float(cash_flow_plan.get("recurring_safe_extra_payment") or 0.0)
    )
    safe_before_floor = (
        baseline_extra
        + essential_delta
        - proposed_discretionary
        - utility_reserve_increment
    )
    safe_extra = max(0.0, safe_before_floor)
    allocation_percent = min(100.0, max(0.0, float(debt_allocation_percent)))
    requested_extra = (
        max(0.0, float(monthly_debt_extra))
        if monthly_debt_extra is not None
        else safe_extra * allocation_percent / 100.0
    )
    minimum_total = sum(
        max(0.0, float(account.get("minimum_payment") or 0.0))
        for account in analysis.get("accounts") or []
        if account.get("type") == "credit"
    )
    proposed_income = (
        suggest_extra_income(analysis) if extra_income is None else extra_income
    )
    streams, extra_payments, uses_estimated = _income_occurrences(proposed_income, today)
    plan = payoff_from_snapshot(
        analysis,
        goals,
        monthly_budget=minimum_total + requested_extra,
        extra_payments_by_month=extra_payments,
        start=today,
    )
    reasons: list[str] = []
    below_floor = [
        row["label"]
        for row in spending
        if row["adjustable"]
        and row["proposed_monthly"] + 0.01 < row["minimum_monthly"]
    ]
    if below_floor:
        reasons.append(
            "These essential budgets are below their safe floor: "
            + ", ".join(below_floor)
            + "."
        )
    if requested_extra > safe_extra + 0.01:
        reasons.append(
            f"The requested ${requested_extra:,.2f} monthly extra payment exceeds "
            f"the calculated safe amount of ${safe_extra:,.2f}."
        )
    if safe_before_floor < -0.01:
        reasons.append(
            f"The proposed monthly spending exceeds available cash by "
            f"${abs(safe_before_floor):,.2f}."
        )
    if plan is None:
        reasons.append("No credit-card balances are available for this plan.")
    elif not plan.get("feasible", True):
        reasons.extend(str(item) for item in plan.get("warnings") or [])
    feasible = not reasons
    status = "feasible" if feasible else "not_feasible"
    if feasible and uses_estimated:
        status = "at_risk"
        reasons.append("The projected timeline depends on estimated extra income.")

    regular_income = 0.0
    scenarios = cash_flow_plan.get("scenarios") or []
    if scenarios:
        regular_income = sum(
            float(item.get("amount") or 0.0)
            for period in scenarios[0].get("pay_periods") or []
            for item in period.get("scheduled_income") or []
            if item.get("category") == "paycheck"
        )
    if regular_income <= 0:
        regular_income = (
            float(analysis.get("total_inflow") or 0.0)
            * 30.0
            / max(1.0, float(analysis.get("period_days") or 30.0))
        )
    minimum_survival = max(
        0.0,
        regular_income - baseline_extra + utility_reserve_increment,
    )
    if plan is not None:
        plan["minimum_payment_total"] = round(minimum_total, 2)
        plan["safe_extra_payment"] = round(safe_extra, 2)
    return {
        "regular_monthly_income": round(regular_income, 2),
        "minimum_survival_budget": round(minimum_survival, 2),
        "baseline_safe_extra": round(baseline_extra, 2),
        "spending_savings": round(spending_savings, 2),
        "safe_monthly_extra": round(safe_extra, 2),
        "planned_monthly_extra": round(requested_extra, 2),
        "monthly_debt_extra": (
            round(max(0.0, float(monthly_debt_extra)), 2)
            if monthly_debt_extra is not None
            else None
        ),
        "debt_allocation_percent": round(allocation_percent, 2),
        "spending": spending,
        "utility_forecast": utility_forecast,
        "extra_income": streams,
        "extra_payments_by_month": {
            month: round(amount, 2) for month, amount in extra_payments.items()
        },
        "feasibility": {
            "status": status,
            "feasible": feasible,
            "depends_on_estimated_income": uses_estimated,
            "reasons": reasons,
        },
        "cash_flow_plan": cash_flow_plan,
        "plan": plan,
    }

"""Deterministic what-if inputs and feasibility for credit-card payoff plans."""
from __future__ import annotations

import calendar
import math
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


def _goal_current_amount(goal: dict[str, Any], analysis: dict[str, Any]) -> float:
    explicit = goal.get("current_amount")
    if explicit is not None:
        return max(0.0, float(explicit))
    milestones = goal.get("milestones") or []
    if milestones:
        return round(
            sum(max(0.0, float(item.get("funded_amount") or 0.0)) for item in milestones),
            2,
        )
    linked = str(goal.get("linked_account") or "").strip().lower()
    if linked:
        for account in analysis.get("accounts") or []:
            if linked in {
                str(account.get("id") or "").strip().lower(),
                str(account.get("name") or "").strip().lower(),
            }:
                return max(0.0, float(account.get("balance") or 0.0))
    return 0.0


def _goal_target_amount(goal: dict[str, Any]) -> float:
    target = goal.get("target_amount")
    if target is not None:
        return max(0.0, float(target))
    return round(
        sum(max(0.0, float(item.get("amount") or 0.0)) for item in goal.get("milestones") or []),
        2,
    )


def _goal_months_remaining(goal: dict[str, Any], today: date) -> int | None:
    target_date = _parse_date(goal.get("target_date"))
    if target_date is None:
        dates = [
            parsed
            for parsed in (
                _parse_date(item.get("due_date")) for item in goal.get("milestones") or []
            )
            if parsed is not None
        ]
        target_date = max(dates) if dates else None
    if target_date is None:
        return None
    return max(1, math.ceil((target_date - today).days / 30.4375))


def _goal_allocation_rows(
    goals: list[dict[str, Any]],
    analysis: dict[str, Any],
    capacity: float,
    debt_allocation_percent: float,
    today: date,
    has_debt: bool,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []
    flexible: list[dict[str, Any]] = []
    for goal in goals:
        kind = str(goal.get("kind") or "savings").lower()
        status = str(goal.get("status") or "active").lower()
        if kind == "debt_payoff" or status != "active":
            continue
        target = _goal_target_amount(goal)
        current = min(target, _goal_current_amount(goal, analysis)) if target else 0.0
        remaining = max(0.0, target - current)
        months = _goal_months_remaining(goal, today)
        deadline_type = str(goal.get("deadline_type") or "soft").lower()
        configured = max(0.0, float(goal.get("monthly_contribution") or 0.0))
        minimum = max(0.0, float(goal.get("minimum_monthly") or 0.0))
        deadline_required = remaining / months if remaining and months else 0.0
        required = max(minimum, deadline_required if deadline_type == "hard" else 0.0)
        desired = max(configured, minimum, required)
        row = {
            "goal_id": str(goal.get("id") or ""),
            "name": str(goal.get("name") or "Goal"),
            "kind": kind,
            "priority": min(5, max(1, int(goal.get("priority") or 3))),
            "horizon": str(goal.get("horizon") or "mid"),
            "deadline_type": deadline_type,
            "target_amount": round(target, 2),
            "current_amount": round(current, 2),
            "remaining": round(remaining, 2),
            "target_date": (
                str(goal.get("target_date")) if goal.get("target_date") else None
            ),
            "required_monthly": round(required, 2),
            "desired_monthly": round(desired, 2),
            "planned_monthly": 0.0,
            "projected_completion_date": None,
            "on_track": None,
        }
        (hard if deadline_type == "hard" else flexible).append(row)

    available = max(0.0, capacity)
    hard.sort(key=lambda row: (row["target_date"] or "9999-12-31", row["priority"]))
    for row in hard:
        planned = min(available, row["desired_monthly"])
        row["planned_monthly"] = round(planned, 2)
        available -= planned
        rows.append(row)

    debt_extra = available * min(100.0, max(0.0, debt_allocation_percent)) / 100.0
    flexible_capacity = available - debt_extra
    flexible.sort(key=lambda row: (row["priority"], row["target_date"] or "9999-12-31"))
    for row in flexible:
        planned = min(flexible_capacity, row["desired_monthly"])
        row["planned_monthly"] = round(planned, 2)
        flexible_capacity -= planned
        rows.append(row)
    if has_debt:
        debt_extra += flexible_capacity
    else:
        flexible_capacity += debt_extra
        debt_extra = 0.0
        for row in flexible:
            if flexible_capacity <= 0:
                break
            remaining_desired = max(0.0, row["desired_monthly"] - row["planned_monthly"])
            addition = min(flexible_capacity, remaining_desired)
            row["planned_monthly"] = round(row["planned_monthly"] + addition, 2)
            flexible_capacity -= addition

    for row in rows:
        planned = row["planned_monthly"]
        remaining = row["remaining"]
        if planned > 0 and remaining > 0:
            completion_months = max(1, math.ceil(remaining / planned))
            row["projected_completion_date"] = _add_months(
                today, completion_months
            ).isoformat()
        if row["deadline_type"] == "hard":
            row["on_track"] = planned + 0.01 >= row["required_monthly"]
    return rows, round(debt_extra, 2)


def _spending_rows(
    analysis: dict[str, Any],
    adjustments: dict[str, float],
    adjustment_reasons: dict[str, str],
    budget_baseline: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    period_days = max(1.0, float(analysis.get("period_days") or 30.0))
    baseline_by_category: dict[str, float] = {}
    for item in budget_baseline or []:
        if not item.get("active", True):
            continue
        category = str(item.get("category") or "")
        baseline_by_category[category] = baseline_by_category.get(category, 0.0) + max(
            0.0, float(item.get("monthly_amount") or 0.0)
        )
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
                if bucket_name == "mandatory" and key in baseline_by_category:
                    current = baseline_by_category[key]
                adjustable = (
                    key not in _UTILITY_SUBCATEGORIES
                    and (
                        bucket_name == "discretionary"
                        or key in _VARIABLE_ESSENTIALS
                    )
                )
                override_allowed = key not in _UTILITY_SUBCATEGORIES
                minimum = current * 0.7 if key in _VARIABLE_ESSENTIALS else 0.0
                default = current
                if adjustable and bucket_name == "discretionary":
                    default = current * (0.75 if key in _DINING else 0.9)
                proposed = (
                    max(0.0, float(adjustments[key]))
                    if key in adjustments and override_allowed
                    else default
                )
                reason = str(adjustment_reasons.get(key) or "").strip()
                changed = abs(proposed - default) > 0.01
                override_requires_reason = changed and (
                    not adjustable or proposed + 0.01 < minimum
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
                        "override_allowed": override_allowed,
                        "override_reason": reason,
                        "override_requires_reason": override_requires_reason,
                    }
                )
    return rows


def suggest_budget_baseline(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Build editable baseline suggestions from normalized mandatory spending."""
    rows = _spending_rows(analysis, {}, {}, None)
    suggestions: list[dict[str, Any]] = []
    for row in rows:
        if row["bucket"] != "mandatory" or row["current_monthly"] <= 0:
            continue
        key = str(row["key"])
        variable = key in _VARIABLE_ESSENTIALS or key in _SEASONAL_UTILITIES
        suggestions.append(
            {
                "id": f"baseline-{key}",
                "name": str(row["label"]),
                "category": key,
                "kind": "variable" if variable else "fixed",
                "monthly_amount": round(float(row["current_monthly"]), 2),
                "due_day": None,
                "source": "inferred",
                "confidence": "low" if variable else "medium",
                "active": True,
            }
        )
    return suggestions


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
        if status == "estimated" and dates and amount > 0:
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
                "debt_amount_per_occurrence": 0.0,
                "savings_amount_per_occurrence": 0.0,
                "goal_allocations": [],
                "allocation_rationale": [],
                "_all_occurrences": [item.isoformat() for item in dates],
                "occurrences": [item.isoformat() for item in dates[:24]],
            }
        )
    return normalized, payments, uses_estimated


def _project_goal_with_extra_income(
    row: dict[str, Any],
    events: list[tuple[date, float]],
    start: date,
) -> str | None:
    remaining = max(0.0, float(row["remaining"]))
    if remaining <= 0:
        return start.isoformat()
    monthly = max(0.0, float(row["planned_monthly"]))
    for offset in range(121):
        period_start = _add_months(start.replace(day=1), offset)
        period_end = _add_months(period_start, 1)
        if offset > 0:
            remaining -= monthly
        for when, amount in events:
            if period_start <= when < period_end:
                remaining -= amount
                if remaining <= 0:
                    return when.isoformat()
        if remaining <= 0:
            return period_start.isoformat()
    return None


def _allocate_extra_income_to_goals(
    streams: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    start: date,
) -> tuple[float, float, float, dict[str, float]]:
    candidates = [row for row in rows if row["kind"] != "debt_payoff"]
    remaining = {row["goal_id"]: float(row["remaining"]) for row in candidates}
    events: dict[str, list[tuple[date, float]]] = {
        row["goal_id"]: [] for row in candidates
    }
    debt_total = 0.0
    debt_remaining = sum(
        abs(float(account.get("balance") or 0.0))
        for account in analysis.get("accounts") or []
        if account.get("type") == "credit"
    )
    goal_total = 0.0
    unassigned_total = 0.0
    debt_payments: dict[str, float] = {}
    for stream in streams:
        occurrences = [
            when
            for when in (
                _parse_date(item)
                for item in stream.pop("_all_occurrences", stream["occurrences"])
            )
            if when is not None
        ]
        amount_each = float(stream["amount"])
        stream_debt = 0.0
        stream_goal = 0.0
        allocations: dict[str, float] = {}
        stream_unassigned = 0.0
        rationale: list[str] = []
        for when in occurrences:
            available = amount_each
            hard = sorted(
                [
                    row
                    for row in candidates
                    if row["deadline_type"] == "hard"
                    and remaining[row["goal_id"]] > 0
                    and (
                        not row["target_date"]
                        or when <= (_parse_date(row["target_date"]) or when)
                    )
                ],
                key=lambda row: (
                    row["priority"],
                    row["target_date"] or "9999-12-31",
                ),
            )
            for row in hard:
                if available <= 0:
                    break
                goal_id = row["goal_id"]
                target = _parse_date(row["target_date"])
                months_after_event = (
                    max(0, math.ceil((target - when).days / 30.4375))
                    if target
                    else 0
                )
                scheduled_before_deadline = (
                    max(0.0, float(row["planned_monthly"])) * months_after_event
                )
                shortfall = max(
                    0.0,
                    remaining[goal_id] - scheduled_before_deadline,
                )
                amount = min(available, shortfall)
                if amount <= 0:
                    continue
                remaining[goal_id] -= amount
                available -= amount
                allocations[goal_id] = allocations.get(goal_id, 0.0) + amount
                events[goal_id].append((when, amount))
                goal_total += amount
                stream_goal += amount
            if available > 0 and debt_remaining > 0:
                debt_amount = min(available, debt_remaining)
                debt_remaining -= debt_amount
                available -= debt_amount
                stream_debt += debt_amount
                debt_total += debt_amount
                month = when.strftime("%Y-%m")
                debt_payments[month] = debt_payments.get(month, 0.0) + debt_amount
            flexible = sorted(
                [
                    row
                    for row in candidates
                    if row["deadline_type"] != "hard"
                    and remaining[row["goal_id"]] > 0
                ],
                key=lambda row: (
                    row["priority"],
                    row["target_date"] or "9999-12-31",
                ),
            )
            for row in flexible:
                if available <= 0:
                    break
                goal_id = row["goal_id"]
                amount = min(available, remaining[goal_id])
                remaining[goal_id] -= amount
                available -= amount
                allocations[goal_id] = allocations.get(goal_id, 0.0) + amount
                events[goal_id].append((when, amount))
                goal_total += amount
                stream_goal += amount
            stream_unassigned += available
        if stream_goal > 0:
            rationale.append("Protects fixed-date and priority goals.")
        if stream_debt > 0:
            rationale.append(
                "Applies remaining funds to credit cards using the payoff engine's "
                "interest-rate and promotional-expiration ordering."
            )
        if stream_unassigned > 0:
            rationale.append("Leaves funds unassigned after current debts and goals are covered.")
        count = max(1, len(occurrences))
        stream["debt_amount_per_occurrence"] = round(stream_debt / count, 2)
        stream["savings_amount_per_occurrence"] = round(stream_goal / count, 2)
        stream["allocation_rationale"] = rationale
        stream["goal_allocations"] = [
            {
                "goal_id": row["goal_id"],
                "name": row["name"],
                "amount": round(allocations[row["goal_id"]], 2),
            }
            for row in candidates
            if allocations.get(row["goal_id"], 0.0) > 0
        ]
        stream["unassigned_savings"] = round(stream_unassigned, 2)
        unassigned_total += stream_unassigned

    for row in candidates:
        goal_id = row["goal_id"]
        allocated = sum(amount for _, amount in events[goal_id])
        row["extra_income_allocated"] = round(allocated, 2)
        row["remaining_after_extra_income"] = round(remaining[goal_id], 2)
        row["projected_completion_date"] = _project_goal_with_extra_income(
            row, events[goal_id], start
        )
        if row["deadline_type"] == "hard" and row["target_date"]:
            target = _parse_date(row["target_date"])
            projected = _parse_date(row["projected_completion_date"])
            row["on_track"] = bool(target and projected and projected <= target)
    return (
        round(debt_total, 2),
        round(goal_total, 2),
        round(unassigned_total, 2),
        {month: round(amount, 2) for month, amount in debt_payments.items()},
    )


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
    spending_adjustment_reasons: dict[str, str] | None = None,
    budget_baseline: list[dict[str, Any]] | None = None,
    debt_allocation_percent: float = 100.0,
    monthly_debt_extra: float | None = None,
    validate_feasibility: bool = True,
) -> dict[str, Any]:
    """Build an editable proposal and a deterministic payoff feasibility result."""
    spending = _spending_rows(
        analysis,
        spending_adjustments or {},
        spending_adjustment_reasons or {},
        budget_baseline,
    )
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
        if row["override_allowed"] and row["bucket"] == "mandatory"
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
    has_debt = any(
        account.get("type") == "credit"
        and abs(float(account.get("balance") or 0.0)) > 0.01
        for account in analysis.get("accounts") or []
    )
    portfolio_rows, recommended_debt_extra = _goal_allocation_rows(
        goals,
        analysis,
        safe_extra,
        allocation_percent,
        today,
        has_debt,
    )
    requested_extra = (
        max(0.0, float(monthly_debt_extra))
        if monthly_debt_extra is not None
        else recommended_debt_extra
    )
    minimum_total = sum(
        max(0.0, float(account.get("minimum_payment") or 0.0))
        for account in analysis.get("accounts") or []
        if account.get("type") == "credit"
    )
    proposed_income = (
        suggest_extra_income(analysis) if extra_income is None else extra_income
    )
    streams, _, uses_estimated = _income_occurrences(proposed_income, today)
    (
        extra_debt_total,
        extra_goal_total,
        extra_unassigned_total,
        extra_payments,
    ) = _allocate_extra_income_to_goals(
        streams, portfolio_rows, analysis, today
    )
    plan = payoff_from_snapshot(
        analysis,
        goals,
        monthly_budget=minimum_total + requested_extra,
        extra_payments_by_month=extra_payments,
        start=today,
    )
    reasons: list[str] = []
    unexplained_overrides = [
        row["label"]
        for row in spending
        if row["override_requires_reason"] and not row["override_reason"]
    ]
    if unexplained_overrides:
        reasons.append(
            "Explain these overrides before relying on the lower budget: "
            + ", ".join(unexplained_overrides)
            + "."
        )
    non_debt_total = sum(row["planned_monthly"] for row in portfolio_rows)
    if requested_extra + non_debt_total > safe_extra + 0.01:
        if non_debt_total:
            reasons.append(
                f"The requested allocations total ${requested_extra + non_debt_total:,.2f} "
                f"but the calculated safe amount is ${safe_extra:,.2f}."
            )
        else:
            reasons.append(
                f"The requested ${requested_extra:,.2f} monthly extra payment exceeds "
                f"the calculated safe amount of ${safe_extra:,.2f}."
            )
    if safe_before_floor < -0.01:
        reasons.append(
            f"The proposed monthly spending exceeds available cash by "
            f"${abs(safe_before_floor):,.2f}."
        )
    behind_goals = [
        row["name"]
        for row in portfolio_rows
        if row["deadline_type"] == "hard" and row["on_track"] is False
    ]
    if behind_goals:
        reasons.append(
            "The current safe surplus cannot fully fund these hard-deadline goals: "
            + ", ".join(behind_goals)
            + "."
        )
    if plan is None and has_debt:
        reasons.append("No credit-card balances are available for this plan.")
    elif plan is not None and not plan.get("feasible", True):
        reasons.extend(str(item) for item in plan.get("warnings") or [])
    calculated_feasible = not reasons
    status = "unchecked"
    feasible: bool | None = None
    if validate_feasibility:
        feasible = calculated_feasible
        status = "feasible" if feasible else "not_feasible"
        if feasible and uses_estimated:
            status = "at_risk"
            reasons.append("The projected timeline depends on estimated extra income.")
    else:
        reasons = []

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
    direct_survival = float(cash_flow_plan.get("monthly_survival_budget") or 0.0)
    minimum_survival = max(
        0.0,
        (
            direct_survival
            if direct_survival > 0
            else regular_income - baseline_extra - essential_delta
        )
        + utility_reserve_increment,
    )
    if plan is not None:
        plan["minimum_payment_total"] = round(minimum_total, 2)
        plan["safe_extra_payment"] = round(safe_extra, 2)
        portfolio_rows.append(
            {
                "goal_id": "credit-card-payoff",
                "name": "Credit-card payoff",
                "kind": "debt_payoff",
                "priority": min(
                    [
                        min(5, max(1, int(goal.get("priority") or 1)))
                        for goal in goals
                        if str(goal.get("kind") or "") == "debt_payoff"
                    ]
                    or [1]
                ),
                "horizon": "short",
                "deadline_type": "hard"
                if any(card.get("deadline") for card in plan.get("cards") or [])
                else "soft",
                "target_amount": round(
                    sum(float(card.get("starting_balance") or 0.0) for card in plan["cards"]),
                    2,
                ),
                "current_amount": 0.0,
                "remaining": round(
                    sum(float(card.get("starting_balance") or 0.0) for card in plan["cards"]),
                    2,
                ),
                "target_date": None,
                "required_monthly": round(minimum_total, 2),
                "desired_monthly": round(minimum_total + requested_extra, 2),
                "planned_monthly": round(requested_extra, 2),
                "projected_completion_date": (
                    _add_months(today, int(plan.get("months_to_debt_free") or 0)).isoformat()
                    if plan.get("months_to_debt_free")
                    else None
                ),
                "on_track": bool(plan.get("feasible", True)),
            }
        )
    total_allocated = round(non_debt_total + requested_extra, 2)
    portfolio_plan = {
        "safe_monthly_capacity": round(safe_extra, 2),
        "total_allocated": total_allocated,
        "unallocated": round(max(0.0, safe_extra - total_allocated), 2),
        "feasible": feasible,
        "warnings": list(reasons),
        "extra_income_to_debt": extra_debt_total,
        "extra_income_to_goals": extra_goal_total,
        "extra_income_unassigned": extra_unassigned_total,
        "allocations": sorted(
            portfolio_rows,
            key=lambda row: (
                0 if row["deadline_type"] == "hard" else 1,
                row["priority"],
                row["target_date"] or "9999-12-31",
            ),
        ),
    }
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
        "budget_baseline": budget_baseline or suggest_budget_baseline(analysis),
        "survival_budget_breakdown": cash_flow_plan.get(
            "survival_budget_breakdown", []
        ),
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
        "portfolio_plan": portfolio_plan,
    }

"""Deterministic credit-card payoff scheduler.

The chat LLM is given account balances, APRs and promotional rates, but asking it
to compute a *strict* month-by-month payoff schedule is unreliable — the interest
math and deadline juggling need to be exact. This module produces that schedule in
code so the conversational layer can present real numbers instead of inventing
them.

Strategy — promo-aware avalanche with hard deadlines:

1.  Each month, interest accrues per card. A promotional balance accrues at its
    promo APR until its ``end_date``; after that it reverts to the card's standard
    APR. The remaining (non-promo) balance always accrues at the standard APR.
2.  Minimum payments are made on every card first.
3.  Any card with a *deadline* — an explicit target date, or a promo's ``end_date``
    (you want the cheap balance cleared before the rate jumps) — is funded next,
    earliest deadline first, at the straight-line amount needed to clear it in time.
4.  Whatever budget is left goes to the highest-APR balance (avalanche), minimising
    total interest.

Within a card, payments always reduce the highest-APR portion first.

The result is a table of ``{month: {card: payment}}`` plus per-card payoff dates,
total interest, and feasibility warnings when the budget can't meet a deadline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

# Below this residual balance a card is considered paid off (rounding dust).
_EPSILON = 0.005
# Default horizon so an under-funded plan terminates instead of looping forever.
_DEFAULT_HORIZON = 120
_MIN_PAYMENT_FLOOR = 25.0
_MIN_PAYMENT_RATE = 0.01  # 1% of the balance, a common card minimum.


@dataclass
class Promo:
    """A promotional balance on a card: ``balance`` accrues at ``apr`` until
    ``end_date``, then reverts to the card's standard APR."""

    balance: float
    apr: float
    end_date: date | None = None


@dataclass
class Card:
    """A revolving debt to pay down. ``balance`` is the positive amount owed."""

    id: str
    name: str
    balance: float
    apr: float = 0.0  # standard APR as a percent, e.g. 23.49
    promos: list[Promo] = field(default_factory=list)
    # An explicit user deadline to have this card fully paid off by.
    target_date: date | None = None
    # Override the computed minimum payment (else max($25, 1% of balance)).
    min_payment: float | None = None


@dataclass
class _Bucket:
    """A slice of a card's balance that accrues at a single APR at a time."""

    remaining: float
    std_apr: float
    promo_apr: float | None = None
    promo_end: date | None = None

    def rate(self, on: date) -> float:
        """Monthly interest rate for this bucket in the month ending ``on``."""
        if self.promo_apr is not None and (self.promo_end is None or on <= self.promo_end):
            return self.promo_apr / 1200.0
        return self.std_apr / 1200.0

    def effective_apr(self, on: date) -> float:
        return self.rate(on) * 1200.0


@dataclass
class _CardState:
    card: Card
    deadline: date | None
    buckets: list[_Bucket]
    total_interest: float = 0.0
    payoff_month: str | None = None

    @property
    def remaining(self) -> float:
        return sum(b.remaining for b in self.buckets)

    @property
    def paid(self) -> bool:
        return self.remaining <= _EPSILON


def _add_months(d: date, n: int) -> date:
    """The last day of the month ``n`` months after ``d``'s month."""
    total = d.year * 12 + (d.month - 1) + n
    year, month = divmod(total, 12)
    month += 1
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _months_between(a: date, b: date) -> int:
    """Whole months from ``a`` to ``b`` (>= 0)."""
    return max(0, (b.year - a.year) * 12 + (b.month - a.month))


def _card_deadline(card: Card) -> date | None:
    """The date this card must be cleared by: the earliest of the user's target
    date and any promo end date (clearing the promo before the rate reverts)."""
    dates = [d for d in [card.target_date, *[p.end_date for p in card.promos]] if d]
    return min(dates) if dates else None


def _build_buckets(card: Card) -> list[_Bucket]:
    """Split a card into a standard bucket plus one bucket per promo, capped so
    the promo balances never exceed the total owed."""
    promo_total = sum(max(0.0, p.balance) for p in card.promos)
    std = max(0.0, card.balance - promo_total)
    buckets: list[_Bucket] = []
    if std > 0:
        buckets.append(_Bucket(remaining=std, std_apr=card.apr))
    for p in card.promos:
        bal = max(0.0, p.balance)
        if bal <= 0:
            continue
        # If promos over-count the balance, scale them down proportionally.
        if promo_total > card.balance and promo_total > 0:
            bal *= card.balance / promo_total
        buckets.append(
            _Bucket(remaining=bal, std_apr=card.apr, promo_apr=p.apr, promo_end=p.end_date)
        )
    if not buckets:
        buckets.append(_Bucket(remaining=max(0.0, card.balance), std_apr=card.apr))
    return buckets


def _pay_card(state: _CardState, amount: float, on: date) -> float:
    """Apply ``amount`` to a card, highest effective APR first. Returns the amount
    actually applied (may be less if the card is smaller than ``amount``)."""
    applied = 0.0
    for bucket in sorted(state.buckets, key=lambda b: b.effective_apr(on), reverse=True):
        if amount <= _EPSILON:
            break
        pay = min(amount, bucket.remaining)
        bucket.remaining -= pay
        amount -= pay
        applied += pay
    return applied


def _min_payment(state: _CardState) -> float:
    card = state.card
    floor = card.min_payment if card.min_payment is not None else max(
        _MIN_PAYMENT_FLOOR, _MIN_PAYMENT_RATE * state.remaining
    )
    return min(floor, state.remaining)


def build_payoff_plan(
    cards: list[Card],
    monthly_budget: float,
    start: date | None = None,
    horizon_months: int = _DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Compute a strict month-by-month payoff schedule.

    ``monthly_budget`` is the total dollars available each month across all cards.
    Returns a JSON-serialisable dict: ``monthly_budget``, ``feasible``, ``warnings``,
    a per-``card`` summary (payoff month, on-time, total interest), a month-by-month
    ``schedule`` (each with per-card payments/interest/remaining), and totals.
    """
    start = start or date.today()
    states = [
        _CardState(card=c, deadline=_card_deadline(c), buckets=_build_buckets(c))
        for c in cards
        if c.balance > _EPSILON
    ]

    warnings: list[str] = []
    schedule: list[dict[str, Any]] = []
    total_interest = 0.0
    total_paid = 0.0

    for m in range(horizon_months):
        if all(s.paid for s in states):
            break
        on = _add_months(start, m)
        month_label = f"{on.year:04d}-{on.month:02d}"
        active = [s for s in states if not s.paid]

        budget = monthly_budget
        paid_this_month: dict[str, float] = {s.card.id: 0.0 for s in active}
        interest_this_month: dict[str, float] = {s.card.id: 0.0 for s in active}

        # 1) Accrue interest.
        for s in active:
            month_interest = 0.0
            for b in s.buckets:
                interest = b.remaining * b.rate(on)
                b.remaining += interest
                month_interest += interest
            s.total_interest += month_interest
            total_interest += month_interest
            interest_this_month[s.card.id] = month_interest

        # 2) Minimum payments on every card.
        for s in active:
            pay = min(_min_payment(s), budget)
            if pay > 0:
                _pay_card(s, pay, on)
                paid_this_month[s.card.id] += pay
                budget -= pay
        if budget <= _EPSILON and any(
            _min_payment(s) > paid_this_month[s.card.id] + _EPSILON for s in active
        ):
            warnings.append(
                f"{month_label}: budget of ${monthly_budget:,.0f}/mo can't cover the "
                "minimum payments on all cards."
            )

        # 3) Deadline-driven straight-line funding, earliest deadline first.
        deadline_cards = sorted(
            (s for s in active if s.deadline is not None and not s.paid),
            key=lambda s: s.deadline,  # type: ignore[arg-type,return-value]
        )
        for s in deadline_cards:
            if budget <= _EPSILON:
                break
            months_left = max(1, _months_between(on, s.deadline) + 1)  # type: ignore[arg-type]
            need = s.remaining / months_left
            extra = min(max(0.0, need - paid_this_month[s.card.id]), s.remaining, budget)
            if extra > 0:
                applied = _pay_card(s, extra, on)
                paid_this_month[s.card.id] += applied
                budget -= applied

        # 4) Avalanche the rest onto the highest-APR remaining balance.
        while budget > _EPSILON:
            candidates = [s for s in active if not s.paid]
            if not candidates:
                break
            target = max(
                candidates,
                key=lambda s: max(b.effective_apr(on) for b in s.buckets),
            )
            pay = min(budget, target.remaining)
            if pay <= _EPSILON:
                break
            applied = _pay_card(target, pay, on)
            paid_this_month[target.card.id] += applied
            budget -= applied

        # Record payments and detect payoffs.
        rows: list[dict[str, Any]] = []
        month_total = 0.0
        for s in active:
            payment = round(paid_this_month[s.card.id], 2)
            month_total += payment
            total_paid += payment
            if s.paid and s.payoff_month is None:
                s.payoff_month = month_label
            rows.append(
                {
                    "card_id": s.card.id,
                    "name": s.card.name,
                    "payment": payment,
                    "interest": round(interest_this_month.get(s.card.id, 0.0), 2),
                    "remaining": round(max(0.0, s.remaining), 2),
                }
            )
        schedule.append(
            {"month": month_label, "payments": rows, "total_payment": round(month_total, 2)}
        )

    # Per-card summary + feasibility.
    card_summaries: list[dict[str, Any]] = []
    feasible = True
    for s in states:
        on_time = True
        if s.deadline is not None:
            if s.payoff_month is None:
                on_time = False
            else:
                py, pm = (int(x) for x in s.payoff_month.split("-"))
                on_time = date(py, pm, 28) <= _add_months(s.deadline, 0)
        if not on_time:
            feasible = False
            when = s.deadline.isoformat() if s.deadline else "the horizon"
            warnings.append(
                f"{s.card.name} can't be paid off by {when} with "
                f"${monthly_budget:,.0f}/mo — increase the monthly budget or extend the date."
            )
        card_summaries.append(
            {
                "id": s.card.id,
                "name": s.card.name,
                "starting_balance": round(s.card.balance, 2),
                "apr": s.card.apr,
                "deadline": s.deadline.isoformat() if s.deadline else None,
                "payoff_month": s.payoff_month,
                "on_time": on_time,
                "total_interest": round(s.total_interest, 2),
            }
        )

    return {
        "monthly_budget": round(monthly_budget, 2),
        "start_month": f"{start.year:04d}-{start.month:02d}",
        "feasible": feasible,
        "warnings": warnings,
        "cards": card_summaries,
        "schedule": schedule,
        "total_interest": round(total_interest, 2),
        "total_paid": round(total_paid, 2),
        "months_to_debt_free": len(schedule) if all(s.paid for s in states) else None,
    }


def cards_from_accounts(
    accounts: list[dict[str, Any]],
    deadlines: dict[str, date] | None = None,
    only_ids: set[str] | None = None,
) -> list[Card]:
    """Build payoff ``Card`` inputs from snapshot account dicts.

    Considers ``credit``-type accounts with a debt (negative ``balance``). Promo
    end dates become implicit deadlines; ``deadlines`` maps an account id OR
    lower-cased name to an explicit user target date (takes precedence). When
    ``only_ids`` is given, only those account ids are included.
    """
    deadlines = deadlines or {}
    cards: list[Card] = []
    for a in accounts:
        if str(a.get("type")) != "credit":
            continue
        balance = float(a.get("balance") or 0.0)
        owed = -balance if balance < 0 else 0.0
        if owed <= _EPSILON:
            continue
        acc_id = str(a.get("id") or a.get("name"))
        if only_ids is not None and acc_id not in only_ids:
            continue
        name = str(a.get("name") or acc_id)
        target = deadlines.get(acc_id) or deadlines.get(name.lower())
        promos: list[Promo] = []
        for p in a.get("promos") or []:
            end = p.get("end_date")
            promos.append(
                Promo(
                    balance=float(p.get("balance") or 0.0),
                    apr=float(p.get("apr") or 0.0),
                    end_date=date.fromisoformat(end) if isinstance(end, str) and end else end,
                )
            )
        cards.append(
            Card(
                id=acc_id,
                name=name,
                balance=owed,
                apr=float(a.get("apr") or 0.0),
                promos=promos,
                target_date=target,
            )
        )
    return cards


def _match_account(name: str, accounts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the snapshot account whose name best matches ``name`` (case-insensitive
    exact, then substring either way)."""
    target = name.strip().lower()
    if not target:
        return None
    for a in accounts:
        if str(a.get("name") or "").strip().lower() == target:
            return a
    for a in accounts:
        an = str(a.get("name") or "").strip().lower()
        if an and (an in target or target in an):
            return a
    return None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


# Everyday variable-living leaves the payoff should hold money back for before
# throwing the rest of the surplus at debt — "food, gas, tolls" and the like.
# Keyed by the analyzer's leaf subcategory names (see analyzer/categorize.py).
_RESERVE_LEAVES = {"groceries", "fuel", "transit", "dining", "coffee", "delivery"}


def essentials_reserve(analysis: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Estimate a reasonable *monthly* set-aside for everyday essentials.

    Sums recent spend on food (groceries + eating out), gas (fuel) and tolls/
    transit from the analyzer's ``spending_tree``, normalised to a monthly figure
    using the analysis window. Returns ``(monthly_total, breakdown_by_leaf)``.

    This is what the payoff "skill" reserves so a strict debt schedule doesn't
    assume every spare dollar can go to cards — the user still has to eat and
    commute. It is a suggested default the user can override.
    """
    tree = analysis.get("spending_tree") or []
    breakdown: dict[str, float] = {}
    for bucket in tree:
        for cat in bucket.get("categories", []):
            for sub in cat.get("subcategories", []):
                leaf = str(sub.get("subcategory") or "")
                if leaf in _RESERVE_LEAVES:
                    breakdown[leaf] = breakdown.get(leaf, 0.0) + abs(
                        float(sub.get("total") or 0.0)
                    )
    days = float(analysis.get("period_days") or analysis.get("lookback_days") or 30) or 30
    scale = 30.0 / days
    breakdown = {k: round(v * scale, 2) for k, v in breakdown.items()}
    return round(sum(breakdown.values()), 2), breakdown


def payoff_from_snapshot(
    analysis: dict[str, Any],
    goals: list[dict[str, Any]] | None = None,
    monthly_budget: float | None = None,
    start: date | None = None,
    reserve: float | None = None,
) -> dict[str, Any] | None:
    """Build a debt-payoff schedule from a chat snapshot and the user's goals.

    The monthly budget defaults to the surplus (income − spending) minus the
    monthly contributions already earmarked for non-debt goals **and an essentials
    reserve** (food/gas/tolls — see ``essentials_reserve``); pass ``monthly_budget``
    to override the whole amount, or ``reserve`` to override just the set-aside.
    Per-card deadlines come from ``debt_payoff`` goals (their ``target_date`` and
    any milestone ``due_date`` naming a card) as well as promo end dates. Returns
    ``None`` when there are no credit-card debts.
    """
    accounts = analysis.get("accounts") or []
    goals = goals or []

    # Which cards to include and their explicit deadlines, gathered from goals.
    deadlines: dict[str, date] = {}
    only_ids: set[str] = set()
    has_debt_goal = False
    for g in goals:
        if str(g.get("kind")) != "debt_payoff":
            continue
        has_debt_goal = True
        gtarget = _parse_date(g.get("target_date"))
        names = g.get("target_accounts") or []
        for nm in names:
            acc = _match_account(str(nm), accounts)
            if acc is None:
                continue
            acc_id = str(acc.get("id") or acc.get("name"))
            only_ids.add(acc_id)
            if gtarget:
                deadlines[acc_id] = gtarget
        for ms in g.get("milestones") or []:
            due = _parse_date(ms.get("due_date"))
            if not due:
                continue
            acc = _match_account(str(ms.get("name") or ""), accounts)
            if acc is not None:
                acc_id = str(acc.get("id") or acc.get("name"))
                only_ids.add(acc_id)
                # A milestone date is the most specific deadline — it wins.
                deadlines[acc_id] = due

    cards = cards_from_accounts(
        accounts, deadlines=deadlines, only_ids=only_ids or None
    )
    if not cards:
        return None

    # Reserve a reasonable monthly set-aside for everyday essentials before any
    # surplus is committed to cards. Auto-derived from recent spend unless the
    # caller passes an explicit override.
    auto_reserve, reserve_breakdown = essentials_reserve(analysis)
    reserve_auto = reserve is None
    reserve_amount = auto_reserve if reserve is None else max(0.0, float(reserve))

    derived = monthly_budget is None
    if monthly_budget is None:
        surplus = float(analysis.get("total_inflow") or 0.0) - float(
            analysis.get("total_outflow") or 0.0
        )
        earmarked = sum(
            float(g.get("monthly_contribution") or 0.0)
            for g in goals
            if str(g.get("kind")) != "debt_payoff"
        )
        monthly_budget = max(0.0, surplus - earmarked - reserve_amount)

    plan = build_payoff_plan(cards, monthly_budget=monthly_budget, start=start)
    plan["derived_budget"] = derived
    plan["scope"] = "goal_cards" if has_debt_goal and only_ids else "all_cards"
    plan["has_debt_goal"] = has_debt_goal
    plan["essentials_reserve"] = reserve_amount
    plan["essentials_reserve_auto"] = reserve_auto
    plan["essentials_reserve_breakdown"] = reserve_breakdown
    return plan

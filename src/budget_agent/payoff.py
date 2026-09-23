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
from decimal import Decimal
from typing import Any

from .money import HALF_CENT, ZERO, decimal_value, json_money, money, money_up

# Below this residual balance a card is considered paid off (rounding dust).
_EPSILON = HALF_CENT
# Default horizon so an under-funded plan terminates instead of looping forever.
_DEFAULT_HORIZON = 120
_MIN_PAYMENT_FLOOR = Decimal("25.00")
_MIN_PAYMENT_RATE = Decimal("0.01")  # 1% of the balance, a common card minimum.


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

    remaining: Decimal
    std_apr: Decimal
    promo_apr: Decimal | None = None
    promo_end: date | None = None
    payoff_month: str | None = None

    def rate(self, on: date) -> Decimal:
        """Monthly interest rate for this bucket in the month ending ``on``."""
        if self.promo_apr is not None and (self.promo_end is None or on <= self.promo_end):
            return self.promo_apr / Decimal("1200")
        return self.std_apr / Decimal("1200")

    def effective_apr(self, on: date) -> Decimal:
        return self.rate(on) * Decimal("1200")


@dataclass
class _CardState:
    card: Card
    deadline: date | None
    buckets: list[_Bucket]
    total_interest: Decimal = ZERO
    payoff_month: str | None = None

    @property
    def remaining(self) -> Decimal:
        return sum((b.remaining for b in self.buckets), start=ZERO)

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
    card_balance = money(max(0.0, card.balance))
    promo_balances = [money(max(0.0, promo.balance)) for promo in card.promos]
    promo_total = sum(promo_balances, start=ZERO)
    std = max(ZERO, card_balance - promo_total)
    buckets: list[_Bucket] = []
    standard_apr = decimal_value(card.apr)
    if std > ZERO:
        buckets.append(_Bucket(remaining=std, std_apr=standard_apr))
    for promo, promo_balance in zip(
        card.promos,
        promo_balances,
        strict=True,
    ):
        bal = promo_balance
        if bal <= ZERO:
            continue
        # If promos over-count the balance, scale them down proportionally.
        if promo_total > card_balance and promo_total > ZERO:
            bal = money(bal * card_balance / promo_total)
        buckets.append(
            _Bucket(
                remaining=bal,
                std_apr=standard_apr,
                promo_apr=decimal_value(promo.apr),
                promo_end=promo.end_date,
            )
        )
    if not buckets:
        buckets.append(_Bucket(remaining=card_balance, std_apr=standard_apr))
    return buckets


def _pay_card(state: _CardState, amount: Decimal, on: date) -> Decimal:
    """Apply ``amount`` to a card, highest effective APR first. Returns the amount
    actually applied (may be less if the card is smaller than ``amount``)."""
    applied = ZERO
    for bucket in sorted(state.buckets, key=lambda b: b.effective_apr(on), reverse=True):
        if amount <= _EPSILON:
            break
        pay = money(min(amount, bucket.remaining))
        bucket.remaining = money(bucket.remaining - pay)
        if bucket.remaining <= _EPSILON and bucket.payoff_month is None:
            bucket.payoff_month = f"{on.year:04d}-{on.month:02d}"
        amount -= pay
        applied += pay
    return applied


def _pay_bucket(bucket: _Bucket, amount: Decimal, on: date) -> Decimal:
    applied = money(min(max(ZERO, amount), bucket.remaining))
    bucket.remaining = money(bucket.remaining - applied)
    if bucket.remaining <= _EPSILON and bucket.payoff_month is None:
        bucket.payoff_month = f"{on.year:04d}-{on.month:02d}"
    return applied


def _min_payment(state: _CardState) -> Decimal:
    card = state.card
    floor = money(card.min_payment) if card.min_payment is not None else max(
        _MIN_PAYMENT_FLOOR, money(_MIN_PAYMENT_RATE * state.remaining)
    )
    return min(floor, state.remaining)


def build_payoff_plan(
    cards: list[Card],
    monthly_budget: float,
    start: date | None = None,
    horizon_months: int = _DEFAULT_HORIZON,
    initial_extra_payment: float = 0.0,
    extra_payments_by_month: dict[str, float] | None = None,
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
        if money(c.balance) > _EPSILON
    ]

    warnings: list[str] = []
    schedule: list[dict[str, Any]] = []
    total_interest = ZERO
    total_paid = ZERO
    minimum_shortfall = False
    extra_payments_by_month = extra_payments_by_month or {}
    monthly_budget_amount = money(max(ZERO, decimal_value(monthly_budget)))
    initial_extra_amount = money(max(ZERO, decimal_value(initial_extra_payment)))

    for m in range(horizon_months):
        if all(s.paid for s in states):
            break
        on = _add_months(start, m)
        month_label = f"{on.year:04d}-{on.month:02d}"
        active = [s for s in states if not s.paid]
        required_minimums = {s.card.id: _min_payment(s) for s in active}

        dated_extra = max(
            ZERO,
            (initial_extra_amount if m == 0 else ZERO)
            + money(extra_payments_by_month.get(month_label)),
        )
        budget = monthly_budget_amount + dated_extra
        paid_this_month: dict[str, Decimal] = {s.card.id: ZERO for s in active}
        interest_this_month: dict[str, Decimal] = {s.card.id: ZERO for s in active}

        # 1) Accrue interest.
        for s in active:
            month_interest = ZERO
            for b in s.buckets:
                interest = money(b.remaining * b.rate(on))
                b.remaining = money(b.remaining + interest)
                month_interest += interest
            s.total_interest += month_interest
            total_interest += month_interest
            interest_this_month[s.card.id] = month_interest

        # 2) Minimum payments on every card.
        for s in active:
            pay = min(required_minimums[s.card.id], budget)
            if pay > ZERO:
                applied = _pay_card(s, pay, on)
                paid_this_month[s.card.id] += applied
                budget -= applied
        if any(
            required_minimums[s.card.id]
            > paid_this_month[s.card.id] + _EPSILON
            for s in active
        ):
            minimum_shortfall = True
            warnings.append(
                f"{month_label}: budget of ${monthly_budget_amount:,.0f}/mo can't cover the "
                "minimum payments on all cards."
            )

        # 3) Direct dated lump sums to the earliest debt deadlines first.
        promo_buckets = sorted(
            (
                (s, bucket)
                for s in active
                for bucket in s.buckets
                if bucket.promo_end is not None and bucket.remaining > _EPSILON
            ),
            key=lambda item: item[1].promo_end,  # type: ignore[arg-type,return-value]
        )
        deadline_lump = min(dated_extra, budget)
        for s, bucket in promo_buckets:
            if deadline_lump <= _EPSILON:
                break
            applied = _pay_bucket(
                bucket,
                min(deadline_lump, budget),
                on,
            )
            paid_this_month[s.card.id] += applied
            deadline_lump -= applied
            budget -= applied

        explicit_deadline_cards = sorted(
            (s for s in active if s.card.target_date is not None and not s.paid),
            key=lambda s: s.card.target_date,  # type: ignore[arg-type,return-value]
        )
        for s in explicit_deadline_cards:
            if deadline_lump <= _EPSILON:
                break
            applied = _pay_card(s, min(deadline_lump, budget), on)
            paid_this_month[s.card.id] += applied
            deadline_lump -= applied
            budget -= applied

        # 4) Fund remaining promotional balances at a straight-line pace.
        for s, bucket in promo_buckets:
            if budget <= _EPSILON:
                break
            months_left = max(1, _months_between(on, bucket.promo_end) + 1)  # type: ignore[arg-type]
            need = money_up(bucket.remaining / months_left)
            applied = _pay_bucket(bucket, min(need, budget), on)
            paid_this_month[s.card.id] += applied
            budget -= applied

        # 5) Fund explicit whole-card deadlines at a straight-line pace.
        for s in explicit_deadline_cards:
            if budget <= _EPSILON:
                break
            months_left = max(1, _months_between(on, s.card.target_date) + 1)  # type: ignore[arg-type]
            need = money_up(s.remaining / months_left)
            extra = min(
                max(ZERO, need - paid_this_month[s.card.id]),
                s.remaining,
                budget,
            )
            if extra > ZERO:
                applied = _pay_card(s, extra, on)
                paid_this_month[s.card.id] += applied
                budget -= applied

        # 6) Avalanche the rest onto the highest-APR remaining balance.
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
        month_total = ZERO
        for s in active:
            payment = money(paid_this_month[s.card.id])
            month_total += payment
            total_paid += payment
            if s.paid and s.payoff_month is None:
                s.payoff_month = month_label
            rows.append(
                {
                    "card_id": s.card.id,
                    "name": s.card.name,
                    "payment": json_money(payment),
                    "interest": json_money(interest_this_month.get(s.card.id, ZERO)),
                    "remaining": json_money(max(ZERO, s.remaining)),
                }
            )
        schedule.append(
            {
                "month": month_label,
                "payments": rows,
                "total_payment": json_money(month_total),
            }
        )

    # Per-card summary + feasibility.
    card_summaries: list[dict[str, Any]] = []
    feasible = not minimum_shortfall
    for s in states:
        explicit_on_time = True
        if s.card.target_date is not None:
            if s.payoff_month is None:
                explicit_on_time = False
            else:
                py, pm = (int(x) for x in s.payoff_month.split("-"))
                explicit_on_time = date(py, pm, 1) <= s.card.target_date
        promo_on_time = all(
            bucket.payoff_month is not None
            and date(
                int(bucket.payoff_month[:4]),
                int(bucket.payoff_month[5:7]),
                1,
            )
            <= bucket.promo_end
            for bucket in s.buckets
            if bucket.promo_end is not None
        )
        on_time = explicit_on_time and promo_on_time
        if not on_time:
            feasible = False
            when = s.deadline.isoformat() if s.deadline else "the horizon"
            scheduled_extra = sum(
                (max(ZERO, money(amount)) for amount in extra_payments_by_month.values()),
                start=ZERO,
            )
            funding = f"${monthly_budget_amount:,.0f}/mo"
            if scheduled_extra > ZERO:
                funding += f" plus ${scheduled_extra:,.0f} in scheduled extra payments"
            warnings.append(
                f"{s.card.name} can't be paid off by {when} with "
                f"{funding} — increase funding or extend the date."
            )
        card_summaries.append(
            {
                "id": s.card.id,
                "name": s.card.name,
                "starting_balance": json_money(money(s.card.balance)),
                "apr": s.card.apr,
                "deadline": s.deadline.isoformat() if s.deadline else None,
                "payoff_month": s.payoff_month,
                "on_time": on_time,
                "total_interest": json_money(s.total_interest),
            }
        )
    if not all(s.paid for s in states):
        feasible = False
        warnings.append(
            f"The plan does not pay off every card within {horizon_months} months."
        )

    return {
        "monthly_budget": json_money(monthly_budget_amount),
        "start_month": f"{start.year:04d}-{start.month:02d}",
        "feasible": feasible,
        "warnings": warnings,
        "cards": card_summaries,
        "schedule": schedule,
        "total_interest": json_money(total_interest),
        "total_paid": json_money(total_paid),
        "months_to_debt_free": len(schedule) if all(s.paid for s in states) else None,
        "initial_extra_payment": json_money(initial_extra_amount),
        "extra_payments_by_month": {
            month: json_money(max(ZERO, money(amount)))
            for month, amount in extra_payments_by_month.items()
            if decimal_value(amount) > ZERO
        },
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
        balance = decimal_value(a.get("balance"))
        owed = money(-balance) if balance < ZERO else ZERO
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
                    balance=json_money(money(p.get("balance"))),
                    apr=float(p.get("apr") or 0.0),
                    end_date=date.fromisoformat(end) if isinstance(end, str) and end else end,
                )
            )
        cards.append(
            Card(
                id=acc_id,
                name=name,
                balance=json_money(owed),
                apr=float(a.get("apr") or 0.0),
                min_payment=(
                    json_money(money(a.get("minimum_payment")))
                    if decimal_value(a.get("minimum_payment")) > ZERO
                    else None
                ),
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
    amounts: dict[str, Decimal] = {}
    for bucket in tree:
        for cat in bucket.get("categories", []):
            for sub in cat.get("subcategories", []):
                leaf = str(sub.get("subcategory") or "")
                if leaf in _RESERVE_LEAVES:
                    amounts[leaf] = amounts.get(leaf, ZERO) + abs(
                        decimal_value(sub.get("total"))
                    )
    days = decimal_value(
        analysis.get("period_days") or analysis.get("lookback_days") or 30
    )
    scale = Decimal("30") / (days or Decimal("30"))
    breakdown = {
        key: json_money(money(value * scale))
        for key, value in amounts.items()
    }
    total = sum((money(value) for value in breakdown.values()), start=ZERO)
    return json_money(total), breakdown


def payoff_from_snapshot(
    analysis: dict[str, Any],
    goals: list[dict[str, Any]] | None = None,
    monthly_budget: float | None = None,
    start: date | None = None,
    reserve: float | None = None,
    initial_extra_payment: float = 0.0,
    extra_payments_by_month: dict[str, float] | None = None,
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

    cards = cards_from_accounts(accounts, deadlines=deadlines)
    if not cards:
        return None

    # Reserve a reasonable monthly set-aside for everyday essentials before any
    # surplus is committed to cards. Auto-derived from recent spend unless the
    # caller passes an explicit override.
    auto_reserve, reserve_breakdown = essentials_reserve(analysis)
    reserve_auto = reserve is None
    reserve_amount = (
        money(auto_reserve)
        if reserve is None
        else money(max(ZERO, decimal_value(reserve)))
    )

    derived = monthly_budget is None
    if monthly_budget is None:
        surplus = decimal_value(analysis.get("total_inflow")) - decimal_value(
            analysis.get("total_outflow")
        )
        earmarked = sum(
            (
                decimal_value(g.get("monthly_contribution"))
                for g in goals
                if str(g.get("kind")) != "debt_payoff"
            ),
            start=ZERO,
        )
        monthly_budget = money(max(ZERO, surplus - earmarked - reserve_amount))

    plan = build_payoff_plan(
        cards,
        monthly_budget=monthly_budget,
        start=start,
        initial_extra_payment=initial_extra_payment,
        extra_payments_by_month=extra_payments_by_month,
    )
    plan["derived_budget"] = derived
    plan["scope"] = "all_cards"
    plan["has_debt_goal"] = has_debt_goal
    plan["essentials_reserve"] = json_money(reserve_amount)
    plan["essentials_reserve_auto"] = reserve_auto
    plan["essentials_reserve_breakdown"] = reserve_breakdown
    return plan

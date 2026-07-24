from datetime import date

from budget_agent.payoff import (
    Card,
    Promo,
    build_payoff_plan,
    cards_from_accounts,
    essentials_reserve,
    payoff_from_snapshot,
)

START = date(2026, 7, 1)


def test_single_card_pays_off_no_interest():
    plan = build_payoff_plan(
        [Card(id="a", name="Card A", balance=1000.0, apr=0.0)],
        monthly_budget=500.0,
        start=START,
    )
    assert plan["total_interest"] == 0.0
    assert plan["months_to_debt_free"] == 2
    summary = plan["cards"][0]
    assert summary["on_time"] is True
    assert summary["payoff_month"] == "2026-08"
    # Every dollar of the balance is eventually paid.
    assert plan["total_paid"] >= 1000.0


def test_avalanche_targets_highest_apr_first():
    plan = build_payoff_plan(
        [
            Card(id="hi", name="High APR", balance=1000.0, apr=25.0),
            Card(id="lo", name="Low APR", balance=1000.0, apr=10.0),
        ],
        monthly_budget=400.0,
        start=START,
    )
    first = {r["card_id"]: r for r in plan["schedule"][0]["payments"]}
    # After minimums, the surplus goes to the 25% card, so it drops faster.
    assert first["hi"]["payment"] > first["lo"]["payment"]
    assert first["hi"]["remaining"] < first["lo"]["remaining"]


def test_promo_deadline_met_with_sufficient_budget():
    # A 0% promo covering the whole balance, expiring in 3 months, becomes the
    # hard deadline. An ample budget should clear it in time.
    plan = build_payoff_plan(
        [
            Card(
                id="boa",
                name="BoA",
                balance=2000.0,
                apr=23.49,
                promos=[Promo(balance=2000.0, apr=0.0, end_date=date(2026, 10, 14))],
            )
        ],
        monthly_budget=800.0,
        start=START,
    )
    boa = plan["cards"][0]
    assert boa["deadline"] == "2026-10-14"
    assert boa["on_time"] is True
    assert boa["payoff_month"] is not None and boa["payoff_month"] <= "2026-10"
    # A 0% promo balance accrues essentially no interest before the deadline.
    assert boa["total_interest"] < 1.0
    assert plan["feasible"] is True


def test_infeasible_deadline_flagged():
    plan = build_payoff_plan(
        [
            Card(
                id="big",
                name="Big Card",
                balance=5000.0,
                apr=22.0,
                target_date=date(2026, 9, 1),
            )
        ],
        monthly_budget=200.0,
        start=START,
    )
    assert plan["feasible"] is False
    assert plan["cards"][0]["on_time"] is False
    assert any("Big Card" in w for w in plan["warnings"])


def test_cards_from_accounts_filters_and_maps():
    accounts = [
        {"id": "chk", "name": "Checking", "type": "checking", "balance": 3000.0},
        {"id": "sav", "name": "Savings", "type": "savings", "balance": 5000.0},
        {"id": "cc0", "name": "Paid Card", "type": "credit", "balance": 0.0},
        {
            "id": "cc1",
            "name": "Chase",
            "type": "credit",
            "balance": -1500.0,
            "apr": 24.99,
            "promos": [{"apr": 0.0, "balance": 500.0, "end_date": "2026-12-01"}],
        },
    ]
    cards = cards_from_accounts(accounts, deadlines={"chase": date(2026, 7, 31)})
    # Only the credit card with a debt is included.
    assert [c.id for c in cards] == ["cc1"]
    chase = cards[0]
    assert chase.balance == 1500.0
    assert chase.apr == 24.99
    assert chase.promos[0].end_date == date(2026, 12, 1)
    # Explicit deadline (by lower-cased name) takes precedence.
    assert chase.target_date == date(2026, 7, 31)


def test_only_ids_restricts_cards():
    accounts = [
        {"id": "cc1", "name": "Chase", "type": "credit", "balance": -1000.0, "apr": 20.0},
        {"id": "cc2", "name": "BoA", "type": "credit", "balance": -2000.0, "apr": 18.0},
    ]
    cards = cards_from_accounts(accounts, only_ids={"cc2"})
    assert [c.id for c in cards] == ["cc2"]


def _snapshot():
    return {
        "total_inflow": 6000.0,
        "total_outflow": 3600.0,  # surplus 2400/mo
        "accounts": [
            {"id": "chk", "name": "Checking", "type": "checking", "balance": 4000.0},
            {"id": "chase", "name": "Chase", "type": "credit", "balance": -1200.0, "apr": 22.0},
            {
                "id": "boa",
                "name": "BoA",
                "type": "credit",
                "balance": -2000.0,
                "apr": 23.49,
                "promos": [{"apr": 0.0, "balance": 2000.0, "end_date": "2026-10-14"}],
            },
        ],
    }


def test_payoff_from_snapshot_derives_budget_and_deadlines():
    goals = [
        {
            "kind": "debt_payoff",
            "name": "Kill the cards",
            "target_accounts": ["Chase", "BoA"],
            "milestones": [{"name": "Chase paid off", "due_date": "2026-07-31"}],
        },
        {"kind": "savings", "name": "Vacation", "monthly_contribution": 400.0},
    ]
    plan = payoff_from_snapshot(_snapshot(), goals, start=START)
    assert plan is not None
    # Budget derived = surplus 2400 − 400 (non-debt goal) = 2000.
    assert plan["monthly_budget"] == 2000.0
    assert plan["derived_budget"] is True
    assert plan["scope"] == "goal_cards"
    cards = {c["id"]: c for c in plan["cards"]}
    # Chase milestone forces a 2026-07 deadline; BoA promo forces 2026-10-14.
    assert cards["chase"]["deadline"] == "2026-07-31"
    assert cards["boa"]["deadline"] == "2026-10-14"


def test_payoff_from_snapshot_none_without_debt():
    snap = {"accounts": [{"id": "chk", "name": "Checking", "type": "checking", "balance": 100.0}]}
    assert payoff_from_snapshot(snap, []) is None


def test_payoff_budget_override_beats_surplus():
    plan = payoff_from_snapshot(_snapshot(), [], monthly_budget=1500.0, start=START)
    assert plan is not None
    assert plan["monthly_budget"] == 1500.0
    assert plan["derived_budget"] is False
    # No debt goals -> all credit cards are in scope.
    assert plan["scope"] == "all_cards"
    assert {c["id"] for c in plan["cards"]} == {"chase", "boa"}


def _snapshot_with_essentials():
    # A 60-day window whose essentials spend halves to a monthly figure.
    snap = _snapshot()
    snap["period_days"] = 60
    snap["spending_tree"] = [
        {
            "bucket": "mandatory",
            "categories": [
                {
                    "category": "groceries",
                    "subcategories": [{"subcategory": "groceries", "total": 800.0}],
                },
                {
                    "category": "transport",
                    "subcategories": [
                        {"subcategory": "fuel", "total": 400.0},
                        {"subcategory": "transit", "total": 100.0},
                    ],
                },
            ],
        },
        {
            "bucket": "discretionary",
            "categories": [
                {
                    "category": "dining",
                    "subcategories": [
                        {"subcategory": "dining", "total": 300.0},
                        {"subcategory": "video_games", "total": 500.0},
                    ],
                }
            ],
        },
    ]
    return snap


def test_essentials_reserve_normalizes_to_monthly():
    total, breakdown = essentials_reserve(_snapshot_with_essentials())
    # 60-day spend halved: groceries 400, fuel 200, transit 50, dining 150 = 800.
    assert breakdown == {"groceries": 400.0, "fuel": 200.0, "transit": 50.0, "dining": 150.0}
    assert total == 800.0
    # Non-essential leaves (video_games) are not reserved.
    assert "video_games" not in breakdown


def test_payoff_reserves_essentials_before_surplus():
    plan = payoff_from_snapshot(_snapshot_with_essentials(), [], start=START)
    assert plan is not None
    # Surplus 2400 − auto reserve 800 = 1600 to cards.
    assert plan["essentials_reserve"] == 800.0
    assert plan["essentials_reserve_auto"] is True
    assert plan["monthly_budget"] == 1600.0
    assert plan["essentials_reserve_breakdown"]["groceries"] == 400.0


def test_payoff_reserve_override():
    plan = payoff_from_snapshot(_snapshot_with_essentials(), [], reserve=300.0, start=START)
    assert plan is not None
    # Explicit reserve overrides the auto figure: 2400 − 300 = 2100.
    assert plan["essentials_reserve"] == 300.0
    assert plan["essentials_reserve_auto"] is False
    assert plan["monthly_budget"] == 2100.0


def test_payoff_has_debt_goal_flag():
    goals = [{"kind": "debt_payoff", "name": "Cards", "target_accounts": ["Chase", "BoA"]}]
    with_goal = payoff_from_snapshot(_snapshot(), goals, start=START)
    without = payoff_from_snapshot(_snapshot(), [], start=START)
    assert with_goal is not None and without is not None
    assert with_goal["has_debt_goal"] is True
    assert without["has_debt_goal"] is False

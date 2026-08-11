from datetime import date, timedelta

from budget_agent.payoff_scenario import build_payoff_scenario, suggest_extra_income


def _analysis():
    return {
        "period_days": 30,
        "accounts": [
            {
                "id": "card",
                "name": "Rewards Card",
                "type": "credit",
                "balance": -3000,
                "apr": 20,
                "minimum_payment": 75,
            }
        ],
        "spending_tree": [
            {
                "bucket": "mandatory",
                "categories": [
                    {
                        "category": "housing",
                        "subcategories": [
                            {"subcategory": "mortgage", "total": 1500},
                            {"subcategory": "groceries", "total": 600},
                        ],
                    }
                ],
            },
            {
                "bucket": "discretionary",
                "categories": [
                    {
                        "category": "food",
                        "subcategories": [
                            {"subcategory": "dining", "total": 400}
                        ],
                    }
                ],
            },
        ],
    }


def _cash_flow():
    return {
        "recurring_safe_extra_payment": 500,
        "scenarios": [
            {
                "pay_periods": [
                    {
                        "scheduled_income": [
                            {"category": "paycheck", "amount": 4000}
                        ]
                    }
                ]
            }
        ],
        "clarification_questions": [],
    }


def test_scenario_locks_fixed_bills_and_recommends_discretionary_cut():
    result = build_payoff_scenario(_analysis(), _cash_flow(), [])
    by_key = {row["key"]: row for row in result["spending"]}

    assert by_key["mortgage"]["adjustable"] is False
    assert by_key["groceries"]["adjustable"] is True
    assert by_key["groceries"]["minimum_monthly"] == 420
    assert by_key["dining"]["proposed_monthly"] == 300
    assert result["safe_monthly_extra"] == 200
    assert result["plan"]["monthly_budget"] == 275


def test_quarterly_income_is_applied_only_in_scheduled_months():
    first = date.today() + timedelta(days=10)
    result = build_payoff_scenario(
        _analysis(),
        _cash_flow(),
        [],
        extra_income=[
            {
                "name": "SCA",
                "amount": 1200,
                "frequency": "quarterly",
                "first_date": first.isoformat(),
                "status": "confirmed",
                "debt_percent": 50,
            }
        ],
    )
    payments = result["extra_payments_by_month"]
    assert payments[first.strftime("%Y-%m")] == 600
    assert len(payments) > 1


def test_unsafe_user_extra_target_is_not_feasible():
    result = build_payoff_scenario(
        _analysis(),
        _cash_flow(),
        [],
        monthly_debt_extra=2000,
    )
    assert result["feasibility"]["status"] == "not_feasible"
    assert any(
        "exceeds the calculated safe amount" in reason
        for reason in result["feasibility"]["reasons"]
    )


def test_suggestions_exclude_regular_direct_deposits_and_do_not_repeat_single_income():
    today = date.today()
    analysis = {
        "income_tree": [
            {
                "source": "ACME DIRECT DEP",
                "transactions": [
                    {"date": (today - timedelta(days=28)).isoformat(), "amount": 2000},
                    {"date": (today - timedelta(days=14)).isoformat(), "amount": 2000},
                ],
            },
            {
                "source": "Annual Bonus",
                "transactions": [
                    {"date": (today - timedelta(days=30)).isoformat(), "amount": 5000}
                ],
            },
        ]
    }
    suggestions = suggest_extra_income(analysis)
    assert [item["name"] for item in suggestions] == ["Annual Bonus"]
    assert suggestions[0]["frequency"] == "one_time"
    assert suggestions[0]["first_date"] is None

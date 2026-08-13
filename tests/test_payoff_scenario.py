from copy import deepcopy
from datetime import date, timedelta

from fastapi.testclient import TestClient

from budget_agent import service
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


def _month(value: date, offset: int) -> date:
    total = value.year * 12 + value.month - 1 + offset
    year, month_index = divmod(total, 12)
    return date(year, month_index + 1, 15)


def _with_utilities(
    analysis,
    *,
    internet=0,
    cell_phone=0,
    electric=0,
    gas_utility=0,
    water=0,
    bucket="mandatory",
):
    result = deepcopy(analysis)
    values = {
        "internet": internet,
        "cell_phone": cell_phone,
        "electric": electric,
        "gas_utility": gas_utility,
        "water": water,
    }
    result["spending_tree"].append(
        {
            "bucket": bucket,
            "categories": [
                {
                    "category": "utilities",
                    "subcategories": [
                        {"subcategory": key, "total": value}
                        for key, value in values.items()
                        if value
                    ],
                }
            ],
        }
    )
    return result


def _utility_history(transactions):
    return {
        "period_days": 730,
        "spending_tree": [
            {
                "bucket": "mandatory",
                "categories": [
                    {
                        "category": "utilities",
                        "subcategories": [
                            {
                                "subcategory": "electric",
                                "transactions": transactions,
                            }
                        ],
                    }
                ],
            }
        ],
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


def test_utility_forecast_uses_calendar_months_with_full_history():
    today = date.today()
    transactions = [
        {
            "date": _month(today, -offset).isoformat(),
            "amount": -200,
        }
        for offset in range(1, 13)
    ] + [{"date": today.isoformat(), "amount": -5000}]
    result = build_payoff_scenario(
        _with_utilities(_analysis(), internet=80, cell_phone=70, electric=200),
        _cash_flow(),
        [],
        utility_history=_utility_history(transactions),
    )

    forecast = result["utility_forecast"]
    assert forecast["stable_monthly_amount"] == 150
    assert forecast["history_months"] == 12
    assert forecast["confidence"] == "high"
    assert forecast["safety_margin_percentage"] == 10
    assert forecast["level_monthly_seasonal_reserve"] == 200
    assert forecast["recommended_protected_monthly_reserve"] == 220
    assert len(forecast["next_12_month_forecasts"]) == 12
    assert all(
        item["basis"] == "same_calendar_month"
        for item in forecast["next_12_month_forecasts"]
    )


def test_sparse_utility_history_uses_conservative_fallback():
    today = date.today()
    first_target = _month(today, 1)
    second_target = _month(today, 2)
    transactions = [
        {
            "date": first_target.replace(year=first_target.year - 1).isoformat(),
            "amount": -100,
        },
        {
            "date": second_target.replace(year=second_target.year - 1).isoformat(),
            "amount": -300,
        },
        {"date": "not-a-date", "amount": "unknown"},
    ]
    result = build_payoff_scenario(
        _with_utilities(_analysis(), electric=150),
        _cash_flow(),
        [],
        utility_history=_utility_history(transactions),
    )

    forecast = result["utility_forecast"]
    assert forecast["history_months"] == 2
    assert forecast["confidence"] == "low"
    assert forecast["safety_margin_percentage"] == 20
    assert forecast["level_monthly_seasonal_reserve"] == 200
    assert forecast["recommended_protected_monthly_reserve"] == 240
    assert sum(
        item["basis"] == "median_fallback"
        for item in forecast["next_12_month_forecasts"]
    ) == 10


def test_all_utilities_remain_protected_from_user_adjustments():
    analysis = _with_utilities(
        _analysis(),
        internet=80,
        cell_phone=70,
        electric=120,
        gas_utility=60,
        water=40,
        bucket="discretionary",
    )
    result = build_payoff_scenario(
        analysis,
        _cash_flow(),
        [],
        spending_adjustments={
            "internet": 0,
            "cell_phone": 0,
            "electric": 0,
            "gas_utility": 0,
            "water": 0,
        },
    )
    utilities = {
        row["key"]: row
        for row in result["spending"]
        if row["key"]
        in {"internet", "cell_phone", "electric", "gas_utility", "water"}
    }

    assert all(row["adjustable"] is False for row in utilities.values())
    assert all(
        row["proposed_monthly"] == row["current_monthly"]
        for row in utilities.values()
    )


def test_incremental_utility_reserve_reduces_safe_debt_capacity():
    today = date.today()
    baseline = build_payoff_scenario(_analysis(), _cash_flow(), [])
    result = build_payoff_scenario(
        _with_utilities(_analysis(), electric=100),
        _cash_flow(),
        [],
        utility_history=_utility_history(
            [
                {"date": _month(today, -offset).isoformat(), "amount": -200}
                for offset in range(1, 13)
            ]
        ),
    )

    assert result["utility_forecast"]["incremental_monthly_reserve"] == 120
    assert result["safe_monthly_extra"] == baseline["safe_monthly_extra"] - 120
    assert (
        result["minimum_survival_budget"]
        == baseline["minimum_survival_budget"] + 120
    )
    assert result["feasibility"]["status"] == "feasible"
    assert result["plan"]["monthly_budget"] == 155


def test_payoff_scenario_endpoint_fetches_separate_utility_history(monkeypatch):
    history = {"period_days": 730, "spending_tree": []}

    class FakeOrchestrator:
        def __init__(self):
            self.snapshot_days = []

        def snapshot(self, days=30):
            self.snapshot_days.append(days)
            return _analysis() if days == 180 else history

        def cash_flow_plan(self, analysis, windfalls, **kwargs):
            return _cash_flow()

    orchestrator = FakeOrchestrator()
    seen = {}

    def scenario(analysis, cash_flow, goals, **kwargs):
        seen["utility_history"] = kwargs["utility_history"]
        return {"plan": {"schedule": []}, "feasibility": {"status": "feasible"}}

    monkeypatch.setattr(service, "_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(service, "build_payoff_scenario", scenario)

    response = TestClient(service.app).post("/payoff-scenario", json={})

    assert response.status_code == 200
    assert orchestrator.snapshot_days == [180, 730]
    assert seen["utility_history"] is history

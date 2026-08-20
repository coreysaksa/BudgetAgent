from copy import deepcopy
from datetime import date, timedelta

from fastapi.testclient import TestClient

from budget_agent import service
from budget_agent.payoff_scenario import (
    build_payoff_scenario,
    reconcile_budget_baseline,
    suggest_budget_baseline,
    suggest_extra_income,
)


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


def test_fixed_mandatory_baseline_uses_observed_monthly_average():
    analysis = _analysis()
    analysis["period_days"] = 180
    mortgage = analysis["spending_tree"][0]["categories"][0]["subcategories"][0]
    mortgage["total"] = 7890
    mortgage["transactions"] = [
        {
            "date": _month(date.today(), -offset).isoformat(),
            "amount": 2630,
            "merchant": "Mortgage servicer",
        }
        for offset in range(3)
    ]

    baseline = suggest_budget_baseline(analysis)
    mortgage_item = next(item for item in baseline if item["category"] == "mortgage")

    assert mortgage_item["monthly_amount"] == 2630
    assert mortgage_item["confidence"] == "high"


def test_fixed_mandatory_uses_arithmetic_average_of_observed_months():
    analysis = _analysis()
    analysis["period_days"] = 180
    mortgage = analysis["spending_tree"][0]["categories"][0]["subcategories"][0]
    mortgage["total"] = 8700
    mortgage["transactions"] = [
        {
            "date": _month(date.today(), -3).isoformat(),
            "amount": 2200,
            "merchant": "Mortgage servicer",
        },
        {
            "date": _month(date.today(), -2).isoformat(),
            "amount": 2500,
            "merchant": "Mortgage servicer",
        },
        {
            "date": _month(date.today(), -1).isoformat(),
            "amount": 4000,
            "merchant": "Mortgage servicer",
        },
    ]

    baseline = suggest_budget_baseline(analysis)
    mortgage_item = next(item for item in baseline if item["category"] == "mortgage")

    assert mortgage_item["monthly_amount"] == 2900
    assert mortgage_item["confidence"] == "high"


def test_sparse_insurance_asks_for_schedule_without_assuming_it():
    analysis = _analysis()
    analysis["period_days"] = 180
    analysis["spending_tree"][0]["categories"].append(
        {
            "category": "insurance",
            "subcategories": [
                {
                    "subcategory": "insurance",
                    "total": 555,
                    "transactions": [
                        {
                            "date": date.today().isoformat(),
                            "amount": 555,
                            "merchant": "Auto insurer",
                        }
                    ],
                }
            ],
        }
    )

    baseline = suggest_budget_baseline(analysis)
    insurance = next(item for item in baseline if item["category"] == "insurance")

    assert insurance["kind"] == "periodic"
    assert insurance["periodic_amount"] == 555
    assert insurance["frequency_months"] is None
    assert insurance["monthly_amount"] == 0
    assert insurance["review_required"] is True
    assert insurance["confidence"] == "low"
    assert "Confirm how often" in insurance["review_prompt"]


def test_confirmed_periodic_reserve_uses_due_date_and_reserved_balance():
    analysis = _analysis()
    today = date.today()
    due = _month(today, 3)
    baseline = [
        {
            "id": "insurance",
            "name": "Car insurance",
            "category": "insurance",
            "kind": "periodic",
            "monthly_amount": 0,
            "periodic_amount": 555,
            "frequency_months": 6,
            "next_due_date": due.isoformat(),
            "reserved_balance": 255,
            "source": "confirmed",
            "confidence": "high",
            "active": True,
        }
    ]

    reconciled = reconcile_budget_baseline(analysis, baseline)

    assert reconciled[0]["monthly_amount"] == 100
    assert reconciled[0]["periodic_amount"] == 555


def test_sparse_vehicle_tax_asks_for_user_amount_and_due_date():
    analysis = _analysis()
    analysis["period_days"] = 180
    analysis["spending_tree"][0]["categories"].append(
        {
            "category": "transport",
            "subcategories": [
                {
                    "subcategory": "vehicle_property_tax",
                    "total": 450,
                    "transactions": [
                        {
                            "date": date.today().isoformat(),
                            "amount": 450,
                            "merchant": "County tax",
                        }
                    ],
                }
            ],
        }
    )

    baseline = suggest_budget_baseline(analysis)
    tax = next(
        item for item in baseline if item["category"] == "vehicle_property_tax"
    )

    assert tax["kind"] == "periodic"
    assert tax["monthly_amount"] == 0
    assert tax["frequency_months"] is None
    assert "amount and due date" in tax["review_prompt"]


def test_regular_monthly_home_maintenance_stays_in_observed_budget():
    analysis = _analysis()
    analysis["period_days"] = 180
    analysis["spending_tree"][0]["categories"].append(
        {
            "category": "housing",
            "subcategories": [
                {
                    "subcategory": "house_maintenance",
                    "total": 600,
                    "transactions": [
                        {
                            "date": _month(date.today(), -offset).isoformat(),
                            "amount": 120,
                            "merchant": "Maintenance provider",
                        }
                        for offset in range(5)
                    ],
                }
            ],
        }
    )

    baseline = suggest_budget_baseline(analysis)
    maintenance = next(
        item for item in baseline if item["category"] == "house_maintenance"
    )

    assert maintenance["kind"] == "fixed"
    assert maintenance["monthly_amount"] == 120
    assert maintenance["review_required"] is False


def test_non_periodic_mandatory_overrides_are_ignored():
    analysis = _analysis()
    inferred = suggest_budget_baseline(analysis)
    mortgage = next(item for item in inferred if item["category"] == "mortgage")
    override = {
        **mortgage,
        "monthly_amount": 1000,
        "source": "confirmed",
    }

    reconciled = reconcile_budget_baseline(analysis, [override])
    reconciled_mortgage = next(
        item for item in reconciled if item["category"] == "mortgage"
    )
    assert reconciled_mortgage["monthly_amount"] == mortgage["monthly_amount"]
    assert reconciled_mortgage["source"] == "inferred"


def test_mortgage_debt_service_cash_outflow_becomes_baseline_item():
    analysis = _analysis()
    analysis["period_days"] = 180
    analysis["spending_tree"][0]["categories"][0]["subcategories"] = [
        item
        for item in analysis["spending_tree"][0]["categories"][0]["subcategories"]
        if item["subcategory"] != "mortgage"
    ]
    analysis["debt_service_outflows"] = [
        {
            "date": _month(date.today(), -offset).isoformat(),
            "amount": 2630,
            "category": "mortgage payment",
        }
        for offset in range(3)
    ]

    baseline = suggest_budget_baseline(analysis)
    mortgage_item = next(item for item in baseline if item["category"] == "mortgage")

    assert mortgage_item["monthly_amount"] == 2630
    assert mortgage_item["kind"] == "fixed"


def test_multiple_mortgages_roll_up_to_total_observed_housing_payment():
    analysis = _analysis()
    analysis["period_days"] = 180
    analysis["spending_tree"][0]["categories"][0]["subcategories"] = [
        item
        for item in analysis["spending_tree"][0]["categories"][0]["subcategories"]
        if item["subcategory"] != "mortgage"
    ]
    analysis["debt_service_outflows"] = [
        {
            "date": _month(date.today(), -offset).isoformat(),
            "amount": amount,
            "category": "mortgage payment",
            "merchant": merchant,
        }
        for offset in range(3)
        for merchant, amount in (
            ("Rocket Mortgage", 1844.35),
            ("Shellpoint Mortgage Servicing", 785.96),
        )
    ]

    baseline = suggest_budget_baseline(analysis)
    mortgage = next(item for item in baseline if item["category"] == "mortgage")

    assert mortgage["name"] == "Mortgage"
    assert mortgage["monthly_amount"] == 2630.31


def test_same_loan_servicer_description_variants_roll_up():
    analysis = _analysis()
    analysis["debt_service_outflows"] = [
        {
            "date": _month(date.today(), -1).isoformat(),
            "amount": 79.91,
            "category": "loan payment",
            "merchant": "Goodleap Servici",
        },
        {
            "date": date.today().isoformat(),
            "amount": 79.91,
            "category": "loan payment",
            "merchant": "Goodleap",
        },
    ]

    baseline = suggest_budget_baseline(analysis)
    loans = [item for item in baseline if item["category"] == "loan_payment"]

    assert len(loans) == 1
    assert loans[0]["monthly_amount"] == 79.91


def test_card_payments_and_interest_are_replaced_by_account_minimums():
    analysis = _analysis()
    analysis["spending_tree"][0]["categories"].append(
        {
            "category": "debt",
            "subcategories": [
                {"subcategory": "credit_card_payment", "total": 5000},
                {"subcategory": "credit_card_interest", "total": 150},
            ],
        }
    )

    baseline = suggest_budget_baseline(analysis)

    assert not {
        "credit_card_payment",
        "credit_card_interest",
    } & {item["category"] for item in baseline}


def test_observed_mortgages_replace_legacy_confirmation():
    analysis = _analysis()
    analysis["spending_tree"][0]["categories"][0]["subcategories"] = [
        item
        for item in analysis["spending_tree"][0]["categories"][0]["subcategories"]
        if item["subcategory"] != "mortgage"
    ]
    analysis["debt_service_outflows"] = [
        {
            "date": date.today().isoformat(),
            "amount": amount,
            "category": "mortgage payment",
            "merchant": merchant,
        }
        for merchant, amount in (
            ("Rocket Mortgage", 1844.35),
            ("Shellpoint Mortgage Servicing", 785.96),
        )
    ]
    legacy = [
        {
            "id": "baseline-mortgage",
            "name": "Mortgage",
            "category": "mortgage",
            "kind": "fixed",
            "monthly_amount": 1224,
            "source": "confirmed",
            "confidence": "high",
            "active": True,
        }
    ]

    reconciled = reconcile_budget_baseline(analysis, legacy)

    assert sum(
        item["monthly_amount"]
        for item in reconciled
        if item["category"] == "mortgage"
    ) == 2630.31
    assert all(item["source"] == "inferred" for item in reconciled)


def test_reconcile_refreshes_inferred_and_confirmed_non_periodic_values():
    analysis = _analysis()
    analysis["period_days"] = 180
    mortgage = analysis["spending_tree"][0]["categories"][0]["subcategories"][0]
    mortgage["total"] = 7890
    mortgage["transactions"] = [
        {
            "date": _month(date.today(), -offset).isoformat(),
            "amount": 2630,
        }
        for offset in range(3)
    ]
    stale = [
        {
            "id": "baseline-mortgage",
            "name": "Mortgage",
            "category": "mortgage",
            "kind": "fixed",
            "monthly_amount": 1224,
            "source": "inferred",
            "confidence": "medium",
            "active": True,
        }
    ]

    refreshed = reconcile_budget_baseline(analysis, stale)
    assert next(
        item for item in refreshed if item["category"] == "mortgage"
    )["monthly_amount"] == 2630

    stale[0]["source"] = "confirmed"
    confirmed = reconcile_budget_baseline(analysis, stale)
    assert next(
        item for item in confirmed if item["category"] == "mortgage"
    )["monthly_amount"] == 2630


def test_variable_spending_includes_each_observed_month_from_long_history():
    analysis = _analysis()
    analysis["period_days"] = 180
    dining = analysis["spending_tree"][1]["categories"][0]["subcategories"][0]
    dining["total"] = 1400
    dining["transactions"] = [
        {
            "date": _month(date.today(), -offset).isoformat(),
            "amount": 100,
            "merchant": "Restaurant",
        }
        for offset in range(1, 6)
    ] + [
        {
            "date": date.today().isoformat(),
            "amount": 900,
            "merchant": "One-time current-month event",
        }
    ]

    result = build_payoff_scenario(analysis, _cash_flow(), [])
    dining_row = next(row for row in result["spending"] if row["key"] == "dining")

    assert dining_row["current_monthly"] == 233.33
    assert dining_row["proposed_monthly"] == 175
    assert dining_row["estimate_confidence"] == "high"


def test_sparse_discretionary_spending_does_not_average_in_zero_months():
    analysis = _analysis()
    analysis["period_days"] = 180
    dining = analysis["spending_tree"][1]["categories"][0]["subcategories"][0]
    dining["total"] = 300
    dining["transactions"] = [
        {
            "date": _month(date.today(), -offset).isoformat(),
            "amount": 100,
            "merchant": "Restaurant",
        }
        for offset in (1, 3, 5)
    ]

    result = build_payoff_scenario(analysis, _cash_flow(), [])
    dining_row = next(row for row in result["spending"] if row["key"] == "dining")

    assert dining_row["current_monthly"] == 100
    assert dining_row["proposed_monthly"] == 75
    assert dining_row["estimate_confidence"] == "high"


def test_single_observed_utility_month_uses_that_month():
    analysis = _analysis()
    analysis["period_days"] = 180
    analysis["spending_tree"][0]["categories"].append(
        {
            "category": "utilities",
            "subcategories": [
                {
                    "subcategory": "electric",
                    "total": 185,
                    "transactions": [
                        {
                            "date": _month(date.today(), -2).isoformat(),
                            "amount": 185,
                            "merchant": "Electric utility",
                        }
                    ],
                }
            ],
        }
    )

    result = build_payoff_scenario(analysis, _cash_flow(), [])
    electric = next(row for row in result["spending"] if row["key"] == "electric")

    assert electric["current_monthly"] == 185
    assert electric["estimate_confidence"] == "low"


def test_two_observed_utility_months_are_divided_by_two():
    analysis = _analysis()
    analysis["period_days"] = 180
    analysis["spending_tree"][0]["categories"].append(
        {
            "category": "utilities",
            "subcategories": [
                {
                    "subcategory": "electric",
                    "total": 300,
                    "transactions": [
                        {
                            "date": _month(date.today(), -2).isoformat(),
                            "amount": 100,
                            "merchant": "Electric utility",
                        },
                        {
                            "date": _month(date.today(), -1).isoformat(),
                            "amount": 200,
                            "merchant": "Electric utility",
                        },
                    ],
                }
            ],
        }
    )

    result = build_payoff_scenario(analysis, _cash_flow(), [])
    electric = next(row for row in result["spending"] if row["key"] == "electric")

    assert electric["current_monthly"] == 150
    assert electric["estimate_confidence"] == "medium"


def test_other_spending_requires_review_and_is_not_treated_as_savings():
    analysis = _analysis()
    analysis["spending_tree"][1]["categories"][0]["subcategories"].append(
        {
            "subcategory": "other",
            "total": 600,
            "transactions": [
                {
                    "date": date.today().isoformat(),
                    "amount": 600,
                    "merchant": "Unknown merchant",
                }
            ],
        }
    )

    result = build_payoff_scenario(analysis, _cash_flow(), [])
    other = next(row for row in result["spending"] if row["key"] == "other")

    assert other["review_required"] is True
    assert other["adjustable"] is False
    assert other["proposed_monthly"] == other["current_monthly"]
    assert other["sample_merchants"] == ["Unknown merchant"]
    assert result["spending_savings"] == 100
    assert any(
        "Classify uncategorized spending" in reason
        for reason in result["feasibility"]["reasons"]
    )


def test_scenario_locks_fixed_bills_and_recommends_discretionary_cut():
    result = build_payoff_scenario(_analysis(), _cash_flow(), [])
    by_key = {row["key"]: row for row in result["spending"]}

    assert by_key["mortgage"]["adjustable"] is False
    assert by_key["mortgage"]["override_allowed"] is False
    assert by_key["groceries"]["adjustable"] is True
    assert by_key["groceries"]["minimum_monthly"] == 420
    assert by_key["dining"]["proposed_monthly"] == 300
    assert result["safe_monthly_extra"] == 200
    assert result["plan"]["monthly_budget"] == 275


def test_confirmed_baseline_drives_direct_survival_budget():
    cash_flow = _cash_flow()
    cash_flow["monthly_survival_budget"] = 3200
    cash_flow["survival_budget_breakdown"] = [
        {
            "id": "mortgage",
            "name": "Confirmed mortgage",
            "category": "mortgage",
            "monthly_amount": 2400,
            "source": "confirmed",
            "confidence": "high",
        }
    ]
    baseline = [
        {
            "id": "mortgage",
            "name": "Confirmed mortgage",
            "category": "mortgage",
            "kind": "fixed",
            "monthly_amount": 2400,
            "source": "confirmed",
            "confidence": "high",
            "active": True,
        }
    ]

    result = build_payoff_scenario(
        _analysis(),
        cash_flow,
        [],
        budget_baseline=baseline,
    )

    assert result["minimum_survival_budget"] == 3200
    assert result["budget_baseline"] == baseline
    assert result["survival_budget_breakdown"][0]["name"] == "Confirmed mortgage"


def test_fixed_obligation_adjustment_is_ignored():
    result = build_payoff_scenario(
        _analysis(),
        _cash_flow(),
        [],
        spending_adjustments={"mortgage": 2594},
        spending_adjustment_reasons={
            "mortgage": "Primary mortgage plus second mortgage."
        },
    )
    mortgage = next(row for row in result["spending"] if row["key"] == "mortgage")

    assert mortgage["proposed_monthly"] == mortgage["current_monthly"]
    assert mortgage["override_reason"] == "Primary mortgage plus second mortgage."
    assert result["minimum_survival_budget"] == 3500
    assert not any(
        "Explain these overrides" in reason
        for reason in result["feasibility"]["reasons"]
    )


def test_below_floor_override_requires_explanation():
    analysis = _analysis()
    analysis["spending_tree"][0]["categories"][0]["subcategories"].append(
        {"subcategory": "fuel", "total": 500}
    )

    unexplained = build_payoff_scenario(
        analysis,
        _cash_flow(),
        [],
        spending_adjustments={"fuel": 200},
    )
    explained = build_payoff_scenario(
        analysis,
        _cash_flow(),
        [],
        spending_adjustments={"fuel": 200},
        spending_adjustment_reasons={
            "fuel": "A one-time forgotten auto charge was cancelled."
        },
    )

    assert any(
        "Explain these overrides" in reason
        for reason in unexplained["feasibility"]["reasons"]
    )
    assert not any(
        "Explain these overrides" in reason
        for reason in explained["feasibility"]["reasons"]
    )
    assert explained["minimum_survival_budget"] == 3200


def test_quarterly_income_ignores_legacy_debt_percentage():
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
    assert payments[first.strftime("%Y-%m")] == 1200
    assert len(payments) > 1


def test_missing_card_minimum_requires_confirmation_for_feasibility():
    analysis = _analysis()
    analysis["accounts"][0]["minimum_payment"] = None
    analysis["accounts"][0]["minimum_payment_status"] = "missing"

    result = build_payoff_scenario(
        analysis,
        _cash_flow(),
        [],
        validate_feasibility=True,
    )

    assert result["feasibility"]["status"] == "at_risk"
    assert any(
        "Confirm the minimum payment for: Rewards Card" in reason
        for reason in result["feasibility"]["reasons"]
    )


def test_zero_balance_card_has_no_minimum_or_confirmation_warning():
    analysis = _analysis()
    analysis["accounts"][0]["balance"] = 0
    analysis["accounts"][0]["minimum_payment"] = 75
    analysis["accounts"][0]["minimum_payment_status"] = "missing"

    result = build_payoff_scenario(
        analysis,
        _cash_flow(),
        [],
        validate_feasibility=True,
    )

    assert result["plan"] is None
    assert result["feasibility"]["status"] == "feasible"
    assert not any(
        "Confirm the minimum payment" in reason
        for reason in result["feasibility"]["reasons"]
    )


def test_extra_income_protects_hard_goal_shortfall_then_pays_debt():
    first = date.today() + timedelta(days=10)
    result = build_payoff_scenario(
        _analysis(),
        _cash_flow(),
        [
            {
                "id": "vacation",
                "name": "Wedding vacation",
                "kind": "milestone",
                "target_amount": 2000,
                "target_date": _month(date.today(), 8).isoformat(),
                "deadline_type": "hard",
            }
        ],
        extra_income=[
            {
                "name": "SCA",
                "amount": 1200,
                "frequency": "one_time",
                "first_date": first.isoformat(),
                "status": "confirmed",
                "debt_percent": 50,
            }
        ],
    )

    stream = result["extra_income"][0]
    assert stream["debt_amount_per_occurrence"] == 800
    assert stream["savings_amount_per_occurrence"] == 400
    assert stream["goal_allocations"] == [
        {"goal_id": "vacation", "name": "Wedding vacation", "amount": 400}
    ]
    assert result["portfolio_plan"]["extra_income_to_debt"] == 800
    assert result["portfolio_plan"]["extra_income_to_goals"] == 400
    assert result["portfolio_plan"]["extra_income_unassigned"] == 0


def test_extra_income_protects_earlier_hard_debt_before_savings():
    today = date.today()
    first = today + timedelta(days=10)
    result = build_payoff_scenario(
        _analysis(),
        _cash_flow(),
        [
            {
                "id": "promo",
                "name": "Promo payoff",
                "kind": "debt_payoff",
                "target_amount": 900,
                "target_date": _month(today, 2).isoformat(),
                "deadline_type": "hard",
                "priority": 1,
            },
            {
                "id": "vacation",
                "name": "Wedding vacation",
                "kind": "savings",
                "target_amount": 2000,
                "target_date": _month(today, 8).isoformat(),
                "deadline_type": "hard",
                "priority": 2,
            },
        ],
        extra_income=[
            {
                "name": "Bonus",
                "amount": 1200,
                "frequency": "one_time",
                "first_date": first.isoformat(),
                "status": "confirmed",
            }
        ],
    )

    stream = result["extra_income"][0]
    assert stream["debt_amount_per_occurrence"] == 900
    assert stream["savings_amount_per_occurrence"] == 300
    assert result["extra_payments_by_month"][first.strftime("%Y-%m")] == 900
    assert stream["goal_allocations"][0] == {
        "goal_id": "promo",
        "name": "Promo payoff",
        "amount": 900,
    }


def test_periodic_mandatory_spending_does_not_create_override_warning():
    analysis = _analysis()
    analysis["spending_tree"][0]["categories"][0]["subcategories"].append(
        {
            "subcategory": "car_maintenance",
            "total": 600,
            "transactions": [
                {
                    "date": date.today().isoformat(),
                    "amount": 600,
                    "merchant": "Repair shop",
                }
            ],
        }
    )
    result = build_payoff_scenario(
        analysis,
        _cash_flow(),
        [],
        budget_baseline=[
            {
                "id": "car-maintenance",
                "name": "Car Maintenance",
                "category": "car_maintenance",
                "kind": "periodic",
                "monthly_amount": 0,
                "periodic_amount": 600,
                "frequency_months": None,
                "source": "inferred",
                "confidence": "low",
                "active": True,
            }
        ],
        spending_adjustments={"car_maintenance": 100},
    )

    maintenance = next(
        row for row in result["spending"] if row["key"] == "car_maintenance"
    )
    assert maintenance["override_allowed"] is False
    assert maintenance["proposed_monthly"] == 0
    assert not any(
        "Car Maintenance" in reason
        for reason in result["feasibility"]["reasons"]
    )


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


def test_feasibility_is_unchecked_until_explicitly_requested():
    result = build_payoff_scenario(
        _analysis(),
        _cash_flow(),
        [],
        monthly_debt_extra=2000,
        validate_feasibility=False,
    )

    assert result["feasibility"]["status"] == "unchecked"
    assert result["feasibility"]["feasible"] is None
    assert result["feasibility"]["reasons"] == []
    assert result["portfolio_plan"]["feasible"] is None
    assert result["portfolio_plan"]["warnings"] == []


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

        def snapshot(self, days=30, **kwargs):
            self.snapshot_days.append((days, kwargs))
            return _analysis() if days == 180 else history

        def cash_flow_plan(self, analysis, windfalls, **kwargs):
            return _cash_flow()

    orchestrator = FakeOrchestrator()
    seen = {}

    def scenario(analysis, cash_flow, goals, **kwargs):
        seen["utility_history"] = kwargs["utility_history"]
        seen["validate_feasibility"] = kwargs["validate_feasibility"]
        return {"plan": {"schedule": []}, "feasibility": {"status": "feasible"}}

    monkeypatch.setattr(service, "_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(service, "build_payoff_scenario", scenario)

    response = TestClient(service.app).post(
        "/payoff-scenario", json={"validate_feasibility": False}
    )

    assert response.status_code == 200
    assert orchestrator.snapshot_days == [
        (730, {}),
        (180, {}),
    ]
    assert seen["utility_history"] is history
    assert seen["validate_feasibility"] is False


def test_budget_baseline_endpoint_refreshes_fixed_values(monkeypatch):
    class FakeOrchestrator:
        def snapshot(self, days=30, **kwargs):
            assert days == 180
            return _analysis()

    monkeypatch.setattr(service, "_orchestrator", FakeOrchestrator)

    response = TestClient(service.app).post(
        "/budget-baseline",
        json={
            "budget_baseline": [
                {
                    "id": "stale-mortgage",
                    "name": "Mortgage",
                    "category": "mortgage",
                    "kind": "fixed",
                    "monthly_amount": 1,
                    "source": "confirmed",
                    "confidence": "high",
                    "active": True,
                }
            ]
        },
    )

    assert response.status_code == 200
    mortgage = next(
        item
        for item in response.json()["budget_baseline"]
        if item["category"] == "mortgage"
    )
    assert mortgage["monthly_amount"] == 1500
    assert mortgage["source"] == "inferred"


def test_transaction_model_preserves_analyzer_classification_evidence():
    from budget_agent.models import Transaction

    transaction = Transaction.model_validate(
        {
            "id": "shellpoint",
            "account_id": "checking",
            "date": date.today().isoformat(),
            "amount": -785.96,
            "description": "NEWREZ-SHELLPOIN DES:ACH PMT",
            "merchant": "Shellpoint Mortgage Servicing",
            "category": "custom_essential",
            "bucket": "mandatory",
            "category_group": "housing",
        }
    ).model_dump(mode="json")

    assert transaction["merchant"] == "Shellpoint Mortgage Servicing"
    assert transaction["bucket"] == "mandatory"
    assert transaction["category_group"] == "housing"


def test_portfolio_protects_hard_deadline_then_prioritizes_debt():
    goal_date = _month(date.today(), 6)
    result = build_payoff_scenario(
        _analysis(),
        _cash_flow(),
        [
            {
                "id": "cards",
                "name": "Pay off cards",
                "kind": "debt_payoff",
                "priority": 1,
            },
            {
                "id": "wedding-trip",
                "name": "Wedding vacation",
                "kind": "milestone",
                "priority": 2,
                "horizon": "short",
                "deadline_type": "hard",
                "target_amount": 1200,
                "target_date": goal_date.isoformat(),
            },
            {
                "id": "house",
                "name": "House down payment",
                "kind": "purchase",
                "priority": 3,
                "horizon": "long",
                "monthly_contribution": 50,
            },
        ],
    )

    portfolio = result["portfolio_plan"]
    by_id = {row["goal_id"]: row for row in portfolio["allocations"]}
    assert by_id["wedding-trip"]["planned_monthly"] == 200
    assert by_id["house"]["planned_monthly"] == 0
    assert by_id["credit-card-payoff"]["planned_monthly"] == 0
    assert portfolio["total_allocated"] == 200
    assert portfolio["feasible"] is True


def test_portfolio_reports_unfunded_hard_deadline():
    result = build_payoff_scenario(
        _analysis(),
        _cash_flow(),
        [
            {
                "id": "trip",
                "name": "Fixed trip",
                "kind": "milestone",
                "deadline_type": "hard",
                "target_amount": 1200,
                "target_date": _month(date.today(), 2).isoformat(),
            }
        ],
    )

    assert result["portfolio_plan"]["feasible"] is False
    assert "Fixed trip" in result["portfolio_plan"]["warnings"][0]

"""Tests for the /chat endpoint's snapshot handling: lookback window parsing
and the data_status signal that distinguishes a fetch failure from empty data.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from fastapi.testclient import TestClient

from budget_agent import service
from budget_agent.models import Windfall


class _FakeReasoner:
    """Records the analysis snapshot it was handed and echoes a canned reply."""

    def __init__(self) -> None:
        self.seen_analysis: dict[str, Any] | None = None

    def chat_and_plan(self, message, analysis, history, current_goals):
        self.seen_analysis = analysis
        return {
            "reply": "ok",
            "goals_updated": False,
            "goals": current_goals,
        }


def test_merge_windfalls_deduplicates_structured_and_extracted_inputs():
    item = Windfall(
        name="SCA",
        amount=1200,
        date=date(2026, 8, 20),
        status="confirmed",
    )
    assert service._merge_windfalls([item], [item]) == [item]


class _FakeOrchestrator:
    def __init__(self, snapshot=None, error: Exception | None = None) -> None:
        self._snapshot = snapshot if snapshot is not None else {"accounts": []}
        self._error = error
        self.seen_days: int | None = None
        self.seen_month: str | None = None
        self.cash_flow_txn_counts: list[int] = []

    def snapshot(
        self,
        days: int = 30,
        month: str | None = None,
    ) -> dict[str, Any]:
        self.seen_days = days
        self.seen_month = month
        if self._error is not None:
            raise self._error
        return dict(self._snapshot)

    def cash_flow_plan(self, analysis, windfalls, **kwargs):
        tree = analysis.get("spending_tree") or []
        if tree:
            self.cash_flow_txn_counts.append(
                len(tree[0]["categories"][0]["subcategories"][0]["transactions"])
            )
        safe_extra = 600.0 if windfalls else 400.0
        return {
            "scenarios": [
                {
                    "safe_extra_payment": safe_extra,
                    "pay_periods": [
                        {
                            "scheduled_income": [
                                {
                                    "amount": 1000,
                                    "category": "paycheck",
                                }
                            ],
                            "obligations": [{"amount": 75}],
                            "essential_allowance": 525,
                        }
                    ],
                }
            ],
            "clarification_questions": [],
            "recurring_safe_extra_payment": 400.0,
        }


def _wire(monkeypatch, reasoner, orch) -> TestClient:
    monkeypatch.setattr(service, "build_reasoner", lambda _s: reasoner)
    monkeypatch.setattr(service, "_orchestrator", lambda: orch)
    return TestClient(service.app)


def test_chat_uses_default_30_day_window(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(snapshot={"accounts": [], "total_inflow": 0.0})
    client = _wire(monkeypatch, reasoner, orch)

    resp = client.post("/chat", json={"message": "where can I save?"})

    assert resp.status_code == 200
    assert orch.seen_days == 30
    status = reasoner.seen_analysis["data_status"]
    assert status == {"ok": True, "lookback_days": 30}
    assert reasoner.seen_analysis["lookback_days"] == 30


def test_chat_widens_window_from_message(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(snapshot={"accounts": [], "total_inflow": 0.0})
    client = _wire(monkeypatch, reasoner, orch)

    client.post("/chat", json={"message": "how did I do over the past 6 months?"})

    assert orch.seen_days == 180
    assert reasoner.seen_analysis["data_status"]["lookback_days"] == 180


def test_chat_supplies_sanitized_page_context_to_reasoner(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(snapshot={"accounts": [], "total_inflow": 0.0})
    client = _wire(monkeypatch, reasoner, orch)

    resp = client.post(
        "/chat",
        json={
            "message": "What stands out here?",
            "page_context": {
                "page": "  Monthly spending  ",
                "title": "August overview",
                "route": "/budget/monthly",
                "selected_month": "2026-08",
                "summary": {
                    "selected_category": "  Utilities  ",
                    "transaction_count": 12,
                    "is_filtered": True,
                },
            },
        },
    )

    assert resp.status_code == 200
    assert reasoner.seen_analysis["page_context"] == {
        "page": "Monthly spending",
        "title": "August overview",
        "route": "/budget/monthly",
        "selected_month": "2026-08",
        "summary": {
            "selected_category": "Utilities",
            "transaction_count": 12,
            "is_filtered": True,
        },
    }


def test_chat_widens_lookback_to_include_selected_month(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(snapshot={"accounts": [], "total_inflow": 0.0})
    client = _wire(monkeypatch, reasoner, orch)
    today = date.today()
    total = today.year * 12 + today.month - 1 - 4
    year, month_index = divmod(total, 12)
    selected = date(year, month_index + 1, 1)
    expected = min(730, (today - selected).days + 1)

    resp = client.post(
        "/chat",
        json={
            "message": "How did I do?",
            "page_context": {"selected_month": selected.strftime("%Y-%m")},
        },
    )

    assert resp.status_code == 200
    assert orch.seen_days == expected
    assert reasoner.seen_analysis["lookback_days"] == expected
    assert reasoner.seen_analysis["data_status"]["lookback_days"] == expected


def test_chat_scopes_overview_and_transaction_context_to_selected_month(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(snapshot={"accounts": [], "total_inflow": 0.0})
    client = _wire(monkeypatch, reasoner, orch)

    resp = client.post(
        "/chat",
        json={
            "message": "What changed here?",
            "page_context": {
                "route": "/app/transactions/spending",
                "selected_month": "2026-07",
            },
        },
    )

    assert resp.status_code == 200
    assert orch.seen_month == "2026-07"


def test_chat_surfaces_degraded_data_status_on_snapshot_failure(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(error=RuntimeError("aggregator down"))
    client = _wire(monkeypatch, reasoner, orch)

    resp = client.post("/chat", json={"message": "looking back 60 days"})

    assert resp.status_code == 200
    status = reasoner.seen_analysis["data_status"]
    assert status["ok"] is False
    assert status["lookback_days"] == 60
    assert "aggregator down" in status["error"]
    # A failed snapshot must not masquerade as real (empty) account data.
    assert "accounts" not in reasoner.seen_analysis


def test_chat_does_not_build_payoff_without_explicit_request(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(snapshot={"accounts": [], "total_inflow": 0.0})
    client = _wire(monkeypatch, reasoner, orch)

    body = client.post("/chat", json={"message": "What is my highest card APR?"}).json()

    assert body["payoff_plan_status"] == "none"
    assert "debt_payoff_plan" not in reasoner.seen_analysis


def test_explicit_payoff_request_returns_draft_without_saving_goal(
    monkeypatch,
):
    class DraftReasoner(_FakeReasoner):
        def extract_cash_flow_inputs(self, message, history):
            return {
                "windfalls": [],
                "paychecks": [],
                "necessity_overrides": [],
                "clarifications": [],
            }

        def chat_and_plan(self, message, analysis, history, current_goals):
            self.seen_analysis = analysis
            return {
                "reply": "Here is the draft.",
                "goals_updated": True,
                "goals": [{"name": "Pay off cards", "kind": "debt_payoff"}],
            }

    seen = {}

    def scenario(*args, **kwargs):
        seen["utility_history"] = kwargs["utility_history"]
        return {
            "plan": {
                "schedule": [],
                "months_to_debt_free": 8,
                "monthly_budget": 475.0,
            },
            "feasibility": {"feasible": True, "status": "feasible"},
        }

    monkeypatch.setattr(service, "build_payoff_scenario", scenario)
    reasoner = DraftReasoner()
    orch = _FakeOrchestrator(
        snapshot={
            "accounts": [
                {
                    "id": "card",
                    "name": "Card",
                    "type": "credit",
                    "minimum_payment": 75,
                }
            ]
        }
    )
    client = _wire(monkeypatch, reasoner, orch)

    body = client.post(
        "/chat", json={"message": "Help me plan to pay off my credit cards."}
    ).json()

    assert body["payoff_plan_status"] == "draft"
    assert body["payoff_plan_ready"] is True
    assert body["goals_updated"] is False
    assert body["payoff_plan"]["monthly_budget"] == 475.0
    assert body["payoff_plan"]["safe_extra_payment"] == 400.0
    assert seen["utility_history"]["accounts"][0]["id"] == "card"


def test_old_payoff_request_does_not_trigger_unrelated_message(monkeypatch):
    class ApprovalReasoner(_FakeReasoner):
        def extract_cash_flow_inputs(self, message, history):
            return {
                "windfalls": [],
                "paychecks": [],
                "necessity_overrides": [],
                "clarifications": [],
            }

    monkeypatch.setattr(
        service,
        "build_payoff_scenario",
        lambda *args, **kwargs: {"plan": {"schedule": [], "monthly_budget": 400}},
    )
    client = _wire(
        monkeypatch,
        ApprovalReasoner(),
        _FakeOrchestrator(snapshot={"accounts": []}),
    )

    body = client.post(
        "/chat",
        json={
            "message": "Add a vacation savings goal.",
            "history": [
                {
                    "role": "user",
                    "content": "Help me plan to pay off my credit cards.",
                }
            ],
        },
    ).json()

    assert body["payoff_plan_status"] == "none"


def test_active_payoff_draft_allows_clarification_followup(monkeypatch):
    class FollowupReasoner(_FakeReasoner):
        def extract_cash_flow_inputs(self, message, history):
            return {
                "windfalls": [],
                "paychecks": [],
                "necessity_overrides": [],
                "clarifications": [],
            }

    monkeypatch.setattr(
        service,
        "build_payoff_scenario",
        lambda *args, **kwargs: {"plan": {"schedule": [], "monthly_budget": 400}},
    )
    client = _wire(
        monkeypatch,
        FollowupReasoner(),
        _FakeOrchestrator(snapshot={"accounts": []}),
    )

    body = client.post(
        "/chat",
        json={
            "message": "I get paid on the 1st and 16th.",
            "payoff_plan_active": True,
        },
    ).json()

    assert body["payoff_plan_status"] == "draft"


def test_payoff_math_uses_untrimmed_spending_snapshot(monkeypatch):
    class PlanReasoner(_FakeReasoner):
        def extract_cash_flow_inputs(self, message, history):
            return {
                "windfalls": [],
                "paychecks": [],
                "necessity_overrides": [],
                "clarifications": [],
            }

    transactions = [
        {"date": f"2026-08-{day:02d}", "amount": 10}
        for day in range(1, 11)
    ]
    snapshot = {
        "accounts": [],
        "spending_tree": [
            {
                "bucket": "mandatory",
                "categories": [
                    {
                        "category": "food",
                        "subcategories": [
                            {"subcategory": "groceries", "transactions": transactions}
                        ],
                    }
                ],
            }
        ],
    }
    orch = _FakeOrchestrator(snapshot=snapshot)
    reasoner = PlanReasoner()
    monkeypatch.setattr(
        service,
        "payoff_from_snapshot",
        lambda *args, **kwargs: {
            "schedule": [],
            "monthly_budget": kwargs["monthly_budget"],
        },
    )

    _wire(monkeypatch, reasoner, orch).post(
        "/chat", json={"message": "Create a credit card payoff plan."}
    )

    assert orch.cash_flow_txn_counts == [10, 10]
    trimmed = reasoner.seen_analysis["spending_tree"][0]["categories"][0][
        "subcategories"
    ][0]["transactions"]
    assert len(trimmed) == 8

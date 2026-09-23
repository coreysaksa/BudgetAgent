from __future__ import annotations

from datetime import date
from typing import Any

from budget_agent.chat_workflow import (
    ChatCommand,
    ChatWorkflow,
    selected_month_lookback,
)


class _Reasoner:
    def __init__(self) -> None:
        self.analysis: dict[str, Any] | None = None

    def extract_cash_flow_inputs(self, message, history):
        raise AssertionError("cash-flow extraction should not run")

    def chat_and_plan(self, message, analysis, history, current_goals):
        self.analysis = analysis
        return {
            "reply": "ok",
            "goals_updated": False,
            "goals": current_goals,
        }


class _Orchestrator:
    def __init__(self) -> None:
        self.snapshot_calls: list[tuple[int, str | None]] = []

    def snapshot(self, days=30, month=None):
        self.snapshot_calls.append((days, month))
        return {"accounts": [], "spending_tree": []}

    def cash_flow_plan(self, *args, **kwargs):
        raise AssertionError("cash-flow planning should not run")


def test_workflow_keeps_non_payoff_chat_on_the_base_path():
    reasoner = _Reasoner()
    orchestrator = _Orchestrator()
    workflow = ChatWorkflow(
        orchestrator,
        reasoner,
        lambda *args, **kwargs: {},
        lambda exc: False,
        today=lambda: date(2026, 9, 9),
    )

    result = workflow.execute(
        ChatCommand(
            message="Where can I save money?",
            history=[],
            goals=[],
            windfalls=[],
            extra_income=[],
            budget_baseline=[],
            checking_buffer=250.0,
            payoff_plan_active=False,
            page_context=None,
        )
    )

    assert orchestrator.snapshot_calls == [(30, None)]
    assert reasoner.analysis is not None
    assert reasoner.analysis["data_status"] == {
        "ok": True,
        "lookback_days": 30,
    }
    assert result["payoff_plan_status"] == "none"


def test_selected_month_lookback_uses_injected_date():
    assert selected_month_lookback(
        "2026-07",
        today=date(2026, 9, 9),
    ) == 71

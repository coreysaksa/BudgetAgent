from types import SimpleNamespace

import json

from budget_agent.approval import MoneyAction
from budget_agent.config import Settings
from budget_agent.models import BudgetLine, BudgetPlan
from budget_agent.reasoning import Reasoner, build_reasoner


class _FakeChat:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[dict] = []

        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._reply)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def _plan():
    return BudgetPlan(
        period="2026-07",
        monthly_income=4000.0,
        lines=[BudgetLine(category="food", allocated=300.0)],
        petty_cash_allocation=150.0,
    )


def test_advise_returns_model_text_and_sends_deployment():
    fake = _FakeChat("Save more on dining out.")
    reasoner = Reasoner(fake, deployment="gpt-4o-mini")
    actions = [MoneyAction(kind="petty_cash_topup", amount=150.0,
                           source_account_id="a", dest_account_id="b", reason="top up")]

    out = reasoner.advise({"total_outflow": 500.0}, _plan(), actions)

    assert out == "Save more on dining out."
    call = fake.calls[0]
    assert call["model"] == "gpt-4o-mini"
    # system + user messages present
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user"]
    assert "petty_cash_topup" in call["messages"][1]["content"]


def test_advise_handles_none_actions():
    fake = _FakeChat("ok")
    reasoner = Reasoner(fake, deployment="gpt-4o-mini")
    assert reasoner.advise({}, _plan()) == "ok"


def test_build_reasoner_returns_none_when_unconfigured():
    assert build_reasoner(Settings(azure_openai_endpoint="")) is None


def test_chat_includes_snapshot_and_history():
    fake = _FakeChat("You spent the most on food.")
    reasoner = Reasoner(fake, deployment="gpt-4o-mini")

    out = reasoner.chat(
        "Where can I cut back?",
        analysis={"by_category": [{"category": "food", "total": 500.0}]},
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )

    assert out == "You spent the most on food."
    msgs = fake.calls[0]["messages"]
    roles = [m["role"] for m in msgs]
    # system prompt, system snapshot, prior user+assistant, then the new user message
    assert roles == ["system", "system", "user", "assistant", "user"]
    assert "food" in msgs[1]["content"]
    assert msgs[-1]["content"] == "Where can I cut back?"


def test_chat_without_snapshot_omits_snapshot_message():
    fake = _FakeChat("No accounts connected yet.")
    reasoner = Reasoner(fake, deployment="gpt-4o-mini")

    reasoner.chat("hello", analysis={}, history=None)

    roles = [m["role"] for m in fake.calls[0]["messages"]]
    assert roles == ["system", "user"]


def test_chat_and_plan_returns_updated_goals():
    reply = json.dumps(
        {
            "reply": "Saving $500/mo reaches $10,000 in about 20 months.",
            "goals_updated": True,
            "goals": [
                {"name": "Emergency fund", "target_amount": 10000,
                 "monthly_contribution": 500}
            ],
        }
    )
    fake = _FakeChat(reply)
    reasoner = Reasoner(fake, deployment="gpt-4o-mini")

    out = reasoner.chat_and_plan(
        "I want to save $10k for emergencies",
        analysis={"by_category": []},
        history=None,
        current_goals=[],
    )

    assert out["goals_updated"] is True
    assert out["goals"] == [
        {"name": "Emergency fund", "target_amount": 10000.0,
         "monthly_contribution": 500.0}
    ]
    # JSON mode is requested and current goals are provided as context.
    call = fake.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert "Current goals" in call["messages"][1]["content"]


def test_chat_and_plan_leaves_goals_unchanged_for_questions():
    current = [{"name": "Vacation", "target_amount": 3000, "monthly_contribution": 250}]
    reply = json.dumps({"reply": "You spent most on food.", "goals_updated": False,
                        "goals": current})
    reasoner = Reasoner(_FakeChat(reply), deployment="gpt-4o-mini")

    out = reasoner.chat_and_plan("Where can I cut back?", current_goals=current)

    assert out["goals_updated"] is False
    assert out["goals"] == current


def test_chat_and_plan_handles_non_json_gracefully():
    current = [{"name": "Vacation", "target_amount": 3000}]
    reasoner = Reasoner(_FakeChat("oops not json"), deployment="gpt-4o-mini")

    out = reasoner.chat_and_plan("hi", current_goals=current)

    assert out["goals_updated"] is False
    assert out["goals"] == current
    assert out["reply"] == "oops not json"


def test_chat_and_plan_drops_nameless_goals():
    reply = json.dumps(
        {
            "reply": "ok",
            "goals_updated": True,
            "goals": [
                {"name": "", "target_amount": 100},
                {"name": "Car", "target_amount": "5000", "monthly_contribution": None},
            ],
        }
    )
    reasoner = Reasoner(_FakeChat(reply), deployment="gpt-4o-mini")

    out = reasoner.chat_and_plan("save for a car", current_goals=[])

    assert out["goals"] == [
        {"name": "Car", "target_amount": 5000.0, "monthly_contribution": None}
    ]

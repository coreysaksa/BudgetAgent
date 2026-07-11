import json

import httpx
import pytest

from budget_agent.approval import MoneyAction
from budget_agent.notifications import Notifier, NotificationPayload


def _action(kind: str = "petty_cash_topup", amount: float = 150.0) -> MoneyAction:
    return MoneyAction(
        kind=kind,
        amount=amount,
        source_account_id="chk",
        dest_account_id="pc",
        reason="Fund petty cash for 2026-07 per budget plan.",
    )


# ---------------------------------------------------------------------------
# NotificationPayload
# ---------------------------------------------------------------------------

def test_payload_as_dict_shape():
    payload = NotificationPayload(
        event="actions_proposed",
        period="2026-07",
        actions=[{"kind": "petty_cash_topup", "amount": 150.0}],
    )
    d = payload.as_dict()
    assert d["event"] == "actions_proposed"
    assert d["period"] == "2026-07"
    assert d["actions"][0]["kind"] == "petty_cash_topup"


# ---------------------------------------------------------------------------
# Notifier: no-op when unconfigured
# ---------------------------------------------------------------------------

def test_notifier_is_noop_when_no_url():
    notifier = Notifier(webhook_url="")
    assert not notifier.configured
    # Should not raise even though there is no transport.
    notifier.notify_proposed([_action()], "2026-07")


# ---------------------------------------------------------------------------
# Notifier: sends correct webhook payload
# ---------------------------------------------------------------------------

def test_notifier_posts_to_webhook():
    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200)

    notifier = Notifier(
        webhook_url="http://webhook.example/notify",
        transport=httpx.MockTransport(handler),
    )
    assert notifier.configured

    actions = [_action(kind="petty_cash_topup", amount=200.0)]
    notifier.notify_proposed(actions, "2026-07")

    assert len(received) == 1
    body = received[0]
    assert body["event"] == "actions_proposed"
    assert body["period"] == "2026-07"
    assert len(body["actions"]) == 1
    a = body["actions"][0]
    assert a["kind"] == "petty_cash_topup"
    assert a["amount"] == 200.0
    assert a["source_account_id"] == "chk"
    assert a["dest_account_id"] == "pc"
    assert "petty cash" in a["reason"]


def test_notifier_sends_all_actions():
    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200)

    notifier = Notifier(
        webhook_url="http://webhook.example/notify",
        transport=httpx.MockTransport(handler),
    )
    actions = [
        _action(kind="petty_cash_topup", amount=150.0),
        _action(kind="transfer", amount=500.0),
    ]
    notifier.notify_proposed(actions, "2026-08")

    assert len(received[0]["actions"]) == 2


# ---------------------------------------------------------------------------
# Notifier: raises on non-2xx webhook response
# ---------------------------------------------------------------------------

def test_notifier_raises_on_webhook_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    notifier = Notifier(
        webhook_url="http://webhook.example/notify",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        notifier.notify_proposed([_action()], "2026-07")


# ---------------------------------------------------------------------------
# Orchestrator integration: notifier called during recommend
# ---------------------------------------------------------------------------

def test_orchestrator_recommend_triggers_notification():
    """Notifier is called with proposed actions when recommend() runs."""
    import httpx as _httpx
    from budget_agent.approval import ApprovalPolicy
    from budget_agent.models import Goal
    from budget_agent.orchestrator import Orchestrator
    from budget_agent.tools import AggregatorClient, AnalyzerClient, PlannerClient

    notified: list[dict] = []

    def webhook_handler(request: _httpx.Request) -> _httpx.Response:
        notified.append(json.loads(request.content))
        return _httpx.Response(200)

    def agg_handler(request: _httpx.Request) -> _httpx.Response:
        if request.url.path == "/accounts":
            return _httpx.Response(200, json=[
                {"id": "chk", "name": "Checking", "type": "checking",
                 "balance": 2000.0, "is_petty_cash": False},
                {"id": "pc", "name": "Petty", "type": "checking",
                 "balance": 100.0, "is_petty_cash": True},
            ])
        return _httpx.Response(200, json=[
            {"id": "t1", "account_id": "chk", "date": "2026-07-01",
             "amount": -40.0, "description": "Coffee", "category": "food"},
        ])

    def ana_handler(request: _httpx.Request) -> _httpx.Response:
        return _httpx.Response(200, json={
            "by_category": [{"category": "food", "total": 40.0}],
            "monthly": [{"month": "2026-07"}],
            "total_inflow": 4000.0,
        })

    def plan_handler(request: _httpx.Request) -> _httpx.Response:
        return _httpx.Response(200, json={
            "period": "2026-07",
            "monthly_income": 4000.0,
            "lines": [{"category": "food", "allocated": 300.0}],
            "petty_cash_allocation": 150.0,
        })

    notifier = Notifier(
        webhook_url="http://webhook.example/notify",
        transport=_httpx.MockTransport(webhook_handler),
    )
    orch = Orchestrator(
        aggregator=AggregatorClient("http://agg", transport=_httpx.MockTransport(agg_handler)),
        analyzer=AnalyzerClient("http://ana", transport=_httpx.MockTransport(ana_handler)),
        planner=PlannerClient("http://plan", transport=_httpx.MockTransport(plan_handler)),
        policy=ApprovalPolicy(require_approval=True),
        notifier=notifier,
    )

    rec = orch.recommend(
        [Goal(id="g1", name="Vacation", target_amount=1000.0)],
        source_account_id="chk",
        petty_cash_account_id="pc",
    )

    assert len(rec.proposed_actions) == 1
    assert len(notified) == 1
    body = notified[0]
    assert body["event"] == "actions_proposed"
    assert body["period"] == "2026-07"
    assert body["actions"][0]["kind"] == "petty_cash_topup"


def test_orchestrator_recommend_skips_notification_when_no_actions():
    """Notifier is NOT called when recommend() produces no proposed actions."""
    import httpx as _httpx
    from budget_agent.approval import ApprovalPolicy
    from budget_agent.orchestrator import Orchestrator
    from budget_agent.tools import AggregatorClient, AnalyzerClient, PlannerClient

    call_count = [0]

    def webhook_handler(request: _httpx.Request) -> _httpx.Response:
        call_count[0] += 1
        return _httpx.Response(200)

    def agg_handler(request: _httpx.Request) -> _httpx.Response:
        if request.url.path == "/accounts":
            return _httpx.Response(200, json=[])
        return _httpx.Response(200, json=[])

    def ana_handler(request: _httpx.Request) -> _httpx.Response:
        return _httpx.Response(200, json={
            "by_category": [], "monthly": [{"month": "2026-07"}], "total_inflow": 0.0,
        })

    def plan_handler(request: _httpx.Request) -> _httpx.Response:
        # No petty_cash_allocation → no actions proposed
        return _httpx.Response(200, json={
            "period": "2026-07", "monthly_income": 0.0,
            "lines": [], "petty_cash_allocation": 0.0,
        })

    notifier = Notifier(
        webhook_url="http://webhook.example/notify",
        transport=_httpx.MockTransport(webhook_handler),
    )
    orch = Orchestrator(
        aggregator=AggregatorClient("http://agg", transport=_httpx.MockTransport(agg_handler)),
        analyzer=AnalyzerClient("http://ana", transport=_httpx.MockTransport(ana_handler)),
        planner=PlannerClient("http://plan", transport=_httpx.MockTransport(plan_handler)),
        policy=ApprovalPolicy(require_approval=True),
        notifier=notifier,
    )

    rec = orch.recommend([])
    assert rec.proposed_actions == []
    assert call_count[0] == 0

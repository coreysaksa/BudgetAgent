import httpx

from budget_agent.models import Goal
from budget_agent.tools import AggregatorClient, AnalyzerClient, PlannerClient


def _transport(handler):
    return httpx.MockTransport(handler)


def test_aggregator_get_accounts():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/accounts"
        return httpx.Response(
            200,
            json=[
                {"id": "acc1", "name": "Checking", "type": "checking",
                 "balance": 1000.0, "is_petty_cash": True},
            ],
        )

    client = AggregatorClient("http://agg", transport=_transport(handler))
    accounts = client.get_accounts()
    assert len(accounts) == 1
    assert accounts[0].id == "acc1"
    assert accounts[0].is_petty_cash is True


def test_aggregator_get_transactions_passes_days():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/transactions"
        assert request.url.params.get("days") == "14"
        return httpx.Response(
            200,
            json=[
                {"id": "t1", "account_id": "acc1", "date": "2026-07-01",
                 "amount": -25.0, "description": "Coffee", "category": "food"},
            ],
        )

    client = AggregatorClient("http://agg", transport=_transport(handler))
    txns = client.get_transactions(days=14)
    assert txns[0].amount == -25.0


def test_analyzer_analyze_posts_payload_and_returns_dict():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/analyze"
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"period_days": 30, "total_outflow": 500.0})

    client = AnalyzerClient("http://ana", transport=_transport(handler))
    result = client.analyze([], [], period_days=30, petty_cash_allowance=100.0)
    assert result["total_outflow"] == 500.0
    assert captured["period_days"] == 30
    assert captured["petty_cash_allowance"] == 100.0


def test_analyzer_progress_not_implemented():
    client = AnalyzerClient("http://ana")
    try:
        client.progress(None)  # type: ignore[arg-type]
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_planner_build_plan_derives_fields():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/plan"
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "period": "2026-07",
                "monthly_income": 4000.0,
                "lines": [{"category": "food", "allocated": 300.0}],
                "petty_cash_allocation": 150.0,
                "goal_contributions": {"g1": 100.0},
                "unallocated": 50.0,
            },
        )

    analysis = {
        "by_category": [{"category": "food", "total": 250.0}],
        "monthly": [{"month": "2026-07"}],
        "total_inflow": 4000.0,
    }
    goals = [Goal(id="g1", name="Vacation", target_amount=1000.0)]
    client = PlannerClient("http://plan", transport=_transport(handler))
    plan = client.build_plan(analysis, goals)

    assert plan.period == "2026-07"
    assert plan.monthly_income == 4000.0
    assert plan.petty_cash_allocation == 150.0
    assert captured["period"] == "2026-07"
    assert captured["monthly_income"] == 4000.0
    assert captured["analysis_by_category"] == {"food": 250.0}


def test_planner_build_plan_defaults_period_when_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(request.content)
        assert body["period"] == "unknown"
        return httpx.Response(200, json={"period": "unknown"})

    client = PlannerClient("http://plan", transport=_transport(handler))
    plan = client.build_plan({}, [])
    assert plan.period == "unknown"


def test_client_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = AggregatorClient("http://agg", transport=_transport(handler))
    try:
        client.get_accounts()
        assert False, "expected HTTPStatusError"
    except httpx.HTTPStatusError:
        pass

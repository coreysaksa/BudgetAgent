import httpx

from budget_agent.approval import ApprovalPolicy
from budget_agent.models import Goal
from budget_agent.orchestrator import Orchestrator
from budget_agent.tools import AggregatorClient, AnalyzerClient, PlannerClient


def _client(cls, handler):
    return cls("http://tool", transport=httpx.MockTransport(handler))


def _build(policy=None):
    def agg_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/accounts":
            return httpx.Response(
                200,
                json=[
                    {"id": "chk", "name": "Checking", "type": "checking",
                     "balance": 2000.0, "is_petty_cash": False},
                    {"id": "pc", "name": "Petty", "type": "checking",
                     "balance": 100.0, "is_petty_cash": True},
                    {"id": "cc", "name": "Rewards Card", "type": "credit",
                     "balance": -640.0, "apr": 19.99,
                     "promos": [{"promo_type": "balance_transfer", "apr": 0.0,
                                 "end_date": "2026-12-01", "balance": 500.0}]},
                ],
            )
        return httpx.Response(
            200,
            json=[
                {"id": "t1", "account_id": "chk", "date": "2026-07-01",
                 "amount": -40.0, "description": "Coffee", "category": "food"},
            ],
        )

    def ana_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "by_category": [{"category": "food", "total": 40.0}],
                "monthly": [{"month": "2026-07"}],
                "total_inflow": 4000.0,
            },
        )

    def plan_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "period": "2026-07",
                "monthly_income": 4000.0,
                "lines": [{"category": "food", "allocated": 300.0}],
                "petty_cash_allocation": 150.0,
            },
        )

    return Orchestrator(
        aggregator=_client(AggregatorClient, agg_handler),
        analyzer=_client(AnalyzerClient, ana_handler),
        planner=_client(PlannerClient, plan_handler),
        policy=policy or ApprovalPolicy(require_approval=True),
    )


def test_snapshot_enriches_analysis_with_account_balances_and_apr():
    orch = _build()
    snap = orch.snapshot()

    # Retains the analyzer output...
    assert snap["total_inflow"] == 4000.0
    # ...and adds a per-account summary the chat layer can ground on.
    accounts = {a["name"]: a for a in snap["accounts"]}
    assert accounts["Checking"]["balance"] == 2000.0
    assert accounts["Checking"]["apr"] is None
    card = accounts["Rewards Card"]
    assert card["balance"] == -640.0
    assert card["type"] == "credit"
    assert card["apr"] == 19.99
    # Promotional rates flow through so the chat/plan layer can steer payoff.
    assert card["promos"] == [
        {"promo_type": "balance_transfer", "apr": 0.0,
         "end_date": "2026-12-01", "balance": 500.0}
    ]
    assert accounts["Checking"]["promos"] == []


def test_snapshot_threads_days_to_transactions_and_analyzer():
    import json as _json

    captured: dict[str, str | int | None] = {}

    def agg_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/accounts":
            return httpx.Response(200, json=[])
        captured["txn_days"] = request.url.params.get("days")
        return httpx.Response(200, json=[])

    def ana_handler(request: httpx.Request) -> httpx.Response:
        captured["period_days"] = _json.loads(request.content)["period_days"]
        return httpx.Response(200, json={"total_inflow": 0.0})

    orch = Orchestrator(
        aggregator=_client(AggregatorClient, agg_handler),
        analyzer=_client(AnalyzerClient, ana_handler),
        planner=_client(PlannerClient, lambda r: httpx.Response(200, json={})),
        policy=ApprovalPolicy(require_approval=True),
    )
    orch.snapshot(days=90)

    assert captured["txn_days"] == "90"
    assert captured["period_days"] == 90


def test_recommend_is_read_only_and_proposes_topup():
    orch = _build()
    goals = [Goal(id="g1", name="Vacation", target_amount=1000.0)]
    rec = orch.recommend(goals, source_account_id="chk", petty_cash_account_id="pc")

    assert rec.plan.period == "2026-07"
    assert rec.analysis["total_inflow"] == 4000.0
    assert len(rec.proposed_actions) == 1
    action = rec.proposed_actions[0]
    assert action.kind == "petty_cash_topup"
    assert action.amount == 150.0
    assert action.source_account_id == "chk"
    assert action.dest_account_id == "pc"


def test_execute_dry_run_reports_approval_required():
    orch = _build(ApprovalPolicy(require_approval=True))
    rec = orch.recommend([])
    results = orch.execute(rec.proposed_actions, approvals={}, dry_run=True)

    assert len(results) == 1
    assert results[0].status == "approval_required"
    assert results[0].dry_run is True


def test_execute_dry_run_auto_approves_capped_topup():
    orch = _build(ApprovalPolicy(require_approval=True, auto_topup_cap=200.0))
    rec = orch.recommend([])
    results = orch.execute(rec.proposed_actions, approvals={}, dry_run=True)

    assert results[0].status == "would_execute"


def test_execute_dry_run_rejects_over_guardrail_even_if_approved():
    orch = _build(ApprovalPolicy(require_approval=True, max_action_amount=100.0))
    rec = orch.recommend([])
    results = orch.execute(
        rec.proposed_actions, approvals={"petty_cash_topup": True}, dry_run=True
    )

    assert results[0].status == "rejected_guardrail"
    # The guardrail denial is recorded to the audit log.
    assert any(e.decision == "denied" for e in orch.audit.entries)


def test_execute_live_is_deferred():
    orch = _build()
    rec = orch.recommend([])
    try:
        orch.execute(rec.proposed_actions, approvals={}, dry_run=False)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass

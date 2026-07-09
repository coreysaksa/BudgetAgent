"""HTTP clients for the three budget tools (aggregator, analyzer, planner).

Each tool is an independently deployed Container App exposing a small JSON API. These
clients wrap those endpoints with httpx and translate the payloads to/from the shared
domain models. A ``transport`` hook is provided so tests can inject
``httpx.MockTransport`` without a live server.
"""
from __future__ import annotations

from typing import Any

import httpx

from .models import Account, BudgetPlan, Goal, Transaction

_TIMEOUT = 30.0


class _BaseClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=_TIMEOUT,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "_BaseClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AggregatorClient(_BaseClient):
    """Reads accounts and transactions from the Plaid aggregator (read-only)."""

    def get_accounts(self) -> list[Account]:
        resp = self._client.get("/accounts")
        resp.raise_for_status()
        return [Account.model_validate(item) for item in resp.json()]

    def get_transactions(self, days: int = 30) -> list[Transaction]:
        resp = self._client.get("/transactions", params={"days": days})
        resp.raise_for_status()
        return [Transaction.model_validate(item) for item in resp.json()]


class AnalyzerClient(_BaseClient):
    """Analyzes spending from accounts + transactions."""

    def analyze(
        self,
        accounts: list[Account],
        transactions: list[Transaction],
        period_days: int = 30,
        petty_cash_allowance: float = 0.0,
    ) -> dict[str, Any]:
        payload = {
            "accounts": [a.model_dump(mode="json") for a in accounts],
            "transactions": [t.model_dump(mode="json") for t in transactions],
            "period_days": period_days,
            "petty_cash_allowance": petty_cash_allowance,
        }
        resp = self._client.post("/analyze", json=payload)
        resp.raise_for_status()
        return resp.json()

    def progress(self, plan: BudgetPlan) -> dict[str, Any]:
        # The analyzer does not expose a progress endpoint yet (M5).
        raise NotImplementedError("analyzer /progress is not available yet")


class PlannerClient(_BaseClient):
    """Builds a budget plan from an analysis + goals."""

    def build_plan(self, analysis: dict[str, Any], goals: list[Goal]) -> BudgetPlan:
        by_category = {
            row["category"]: row["total"]
            for row in analysis.get("by_category", [])
        }
        monthly = analysis.get("monthly", [])
        period = monthly[-1]["month"] if monthly else "unknown"
        monthly_income = float(analysis.get("total_inflow", 0.0))
        payload = {
            "period": period,
            "monthly_income": monthly_income,
            "analysis_by_category": by_category,
            "goals": [g.model_dump(mode="json") for g in goals],
        }
        resp = self._client.post("/plan", json=payload)
        resp.raise_for_status()
        return BudgetPlan.model_validate(resp.json())

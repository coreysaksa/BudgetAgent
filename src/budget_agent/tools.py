"""Typed clients for the separate tool services.

Each tool lives in its own repo and is called over HTTP (local dev defaults in .env.example).
These are thin stubs to be fleshed out as the tools are implemented.
"""
from __future__ import annotations

from .models import Account, BudgetPlan, Goal, Transaction


class AggregatorClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def get_accounts(self) -> list[Account]:
        raise NotImplementedError

    def get_transactions(self) -> list[Transaction]:
        raise NotImplementedError


class AnalyzerClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def analyze(self, accounts: list[Account], txns: list[Transaction]):
        raise NotImplementedError

    def progress(self, plan: BudgetPlan):
        raise NotImplementedError


class PlannerClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def build_plan(self, analysis, goals: list[Goal]) -> BudgetPlan:
        raise NotImplementedError

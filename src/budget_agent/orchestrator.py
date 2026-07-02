"""The analyze -> plan -> propose -> approve -> execute -> track state machine."""
from __future__ import annotations

from enum import Enum

from .approval import ApprovalPolicy, MoneyAction
from .models import BudgetPlan, Goal
from .tools import AggregatorClient, AnalyzerClient, PlannerClient


class Phase(str, Enum):
    ANALYZE = "analyze"
    PLAN = "plan"
    PROPOSE = "propose"
    APPROVE = "approve"
    EXECUTE = "execute"
    TRACK = "track"


class Orchestrator:
    def __init__(
        self,
        aggregator: AggregatorClient,
        analyzer: AnalyzerClient,
        planner: PlannerClient,
        policy: ApprovalPolicy,
    ) -> None:
        self.aggregator = aggregator
        self.analyzer = analyzer
        self.planner = planner
        self.policy = policy

    def analyze(self):
        """Pull transactions (read-only) and analyze spending."""
        accounts = self.aggregator.get_accounts()
        txns = self.aggregator.get_transactions()
        return self.analyzer.analyze(accounts, txns)

    def plan(self, analysis, goals: list[Goal]) -> BudgetPlan:
        """Build a budget from goals + analyzed spending."""
        return self.planner.build_plan(analysis, goals)

    def propose(self, plan: BudgetPlan) -> list[MoneyAction]:
        """Recommend money movements to realize the plan (not yet executed)."""
        # Placeholder: derive recommended transfers from the plan.
        return []

    def execute(self, actions: list[MoneyAction], approvals: dict[str, bool]) -> None:
        """Apply approved actions. Money movement passes through the approval gate."""
        for action in actions:
            self.policy.guard(action, human_approved=approvals.get(action.kind, False))
            # TODO: perform the (approved) action via the execution adapter.

    def track(self, plan: BudgetPlan):
        """Fully-automated: recompute progress toward goals and budget lines."""
        return self.analyzer.progress(plan)

"""The analyze -> plan -> propose -> approve -> execute -> track state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .approval import (
    ApprovalPolicy,
    ApprovalRequired,
    ExecutionResult,
    GuardrailViolation,
    MoneyAction,
)
from .audit import AuditLog
from .models import BudgetPlan, Goal
from .tools import AggregatorClient, AnalyzerClient, PlannerClient


class Phase(str, Enum):
    ANALYZE = "analyze"
    PLAN = "plan"
    PROPOSE = "propose"
    APPROVE = "approve"
    EXECUTE = "execute"
    TRACK = "track"


@dataclass
class Recommendation:
    """Read-only output of the recommend flow: analysis + plan + proposed actions."""
    analysis: dict[str, Any]
    plan: BudgetPlan
    proposed_actions: list[MoneyAction] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        aggregator: AggregatorClient,
        analyzer: AnalyzerClient,
        planner: PlannerClient,
        policy: ApprovalPolicy,
        audit: AuditLog | None = None,
    ) -> None:
        self.aggregator = aggregator
        self.analyzer = analyzer
        self.planner = planner
        self.policy = policy
        self.audit = audit or AuditLog()

    def analyze(self):
        """Pull transactions (read-only) and analyze spending."""
        accounts = self.aggregator.get_accounts()
        txns = self.aggregator.get_transactions()
        return self.analyzer.analyze(accounts, txns)

    def plan(self, analysis, goals: list[Goal]) -> BudgetPlan:
        """Build a budget from goals + analyzed spending."""
        return self.planner.build_plan(analysis, goals)

    def propose(
        self,
        plan: BudgetPlan,
        source_account_id: str = "",
        petty_cash_account_id: str = "",
    ) -> list[MoneyAction]:
        """Recommend money movements to realize the plan (not yet executed)."""
        actions: list[MoneyAction] = []
        if plan.petty_cash_allocation > 0:
            actions.append(
                MoneyAction(
                    kind="petty_cash_topup",
                    amount=plan.petty_cash_allocation,
                    source_account_id=source_account_id,
                    dest_account_id=petty_cash_account_id,
                    reason=f"Fund petty cash for {plan.period} per budget plan.",
                )
            )
        return actions

    def recommend(
        self,
        goals: list[Goal],
        source_account_id: str = "",
        petty_cash_account_id: str = "",
    ) -> Recommendation:
        """Read-only mode: analyze -> plan -> propose. Never moves money."""
        analysis = self.analyze()
        plan = self.plan(analysis, goals)
        actions = self.propose(plan, source_account_id, petty_cash_account_id)
        return Recommendation(analysis=analysis, plan=plan, proposed_actions=actions)

    def execute(
        self,
        actions: list[MoneyAction],
        approvals: dict[str, bool],
        dry_run: bool = True,
    ) -> list[ExecutionResult]:
        """Run actions through the approval gate + guardrails.

        Live money movement is deferred (high risk), so only dry-run is supported: each
        action is validated and its would-be outcome is reported without moving money.
        """
        if not dry_run:
            raise NotImplementedError(
                "Live money movement is deferred; only dry_run execution is supported."
            )
        results: list[ExecutionResult] = []
        for action in actions:
            try:
                self.policy.guard(
                    action,
                    human_approved=approvals.get(action.kind, False),
                    audit=self.audit,
                )
                results.append(
                    ExecutionResult(
                        kind=action.kind,
                        amount=action.amount,
                        status="would_execute",
                        dry_run=True,
                        detail="Dry run: approved and validated; no money moved.",
                    )
                )
            except GuardrailViolation as exc:
                results.append(
                    ExecutionResult(
                        kind=action.kind,
                        amount=action.amount,
                        status="rejected_guardrail",
                        dry_run=True,
                        detail=str(exc),
                    )
                )
            except ApprovalRequired as exc:
                results.append(
                    ExecutionResult(
                        kind=action.kind,
                        amount=action.amount,
                        status="approval_required",
                        dry_run=True,
                        detail=str(exc),
                    )
                )
        return results

    def track(self, plan: BudgetPlan):
        """Fully-automated: recompute progress toward goals and budget lines."""
        return self.analyzer.progress(plan)

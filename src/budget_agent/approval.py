"""Human-approval gate for any action that moves money.

Autonomous money movement is high-risk and regulated. By default every money-moving
action must be explicitly approved. Auto-execution is only allowed for actions that a
narrow, opt-in policy marks as safe (e.g. a capped top-up to the petty-cash account).

Guardrails add a hard per-action ceiling (``max_action_amount``): actions above it are
rejected outright, even with human approval. This bounds worst-case blast radius while
live money movement remains deferred (see orchestrator.execute dry-run).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audit import AuditLog


@dataclass
class MoneyAction:
    kind: str            # e.g. "transfer", "petty_cash_topup"
    amount: float
    source_account_id: str
    dest_account_id: str
    reason: str


@dataclass
class ExecutionResult:
    """Outcome of (dry-run) processing a single money action through the gate."""
    kind: str
    amount: float
    status: str          # would_execute | approval_required | rejected_guardrail
    dry_run: bool
    detail: str


class ApprovalRequired(Exception):
    """Raised when an action needs human approval but none was granted."""


class GuardrailViolation(Exception):
    """Raised when an action exceeds a hard guardrail (e.g. the per-action limit)."""


class ApprovalPolicy:
    def __init__(
        self,
        require_approval: bool = True,
        auto_topup_cap: float = 0.0,
        max_action_amount: float = 0.0,
    ) -> None:
        self.require_approval = require_approval
        self.auto_topup_cap = auto_topup_cap
        # Hard ceiling on any single action. 0 disables the limit.
        self.max_action_amount = max_action_amount

    def is_auto_allowed(self, action: MoneyAction) -> bool:
        """Only capped petty-cash top-ups may auto-execute, and only if opted in."""
        if not self.require_approval:
            return True
        return (
            action.kind == "petty_cash_topup"
            and 0 < action.amount <= self.auto_topup_cap
        )

    def guard(
        self,
        action: MoneyAction,
        human_approved: bool = False,
        audit: "AuditLog | None" = None,
    ) -> None:
        # Hard guardrail: over-limit actions are denied regardless of approval.
        if self.max_action_amount and action.amount > self.max_action_amount:
            if audit is not None:
                audit.record(action, "denied")
            raise GuardrailViolation(
                f"Action '{action.kind}' for {action.amount} exceeds the "
                f"per-action limit of {self.max_action_amount}."
            )
        if human_approved:
            if audit is not None:
                audit.record(action, "human_approved")
            return
        if self.is_auto_allowed(action):
            if audit is not None:
                audit.record(action, "auto_approved")
            return
        if audit is not None:
            audit.record(action, "denied")
        raise ApprovalRequired(
            f"Action '{action.kind}' for {action.amount} requires human approval."
        )

"""Human-approval gate for any action that moves money.

Autonomous money movement is high-risk and regulated. By default every money-moving
action must be explicitly approved. Auto-execution is only allowed for actions that a
narrow, opt-in policy marks as safe (e.g. a capped top-up to the petty-cash account).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MoneyAction:
    kind: str            # e.g. "transfer", "petty_cash_topup"
    amount: float
    source_account_id: str
    dest_account_id: str
    reason: str


class ApprovalRequired(Exception):
    """Raised when an action needs human approval but none was granted."""


class ApprovalPolicy:
    def __init__(self, require_approval: bool = True, auto_topup_cap: float = 0.0) -> None:
        self.require_approval = require_approval
        self.auto_topup_cap = auto_topup_cap

    def is_auto_allowed(self, action: MoneyAction) -> bool:
        """Only capped petty-cash top-ups may auto-execute, and only if opted in."""
        if not self.require_approval:
            return True
        return (
            action.kind == "petty_cash_topup"
            and 0 < action.amount <= self.auto_topup_cap
        )

    def guard(self, action: MoneyAction, human_approved: bool = False) -> None:
        if human_approved or self.is_auto_allowed(action):
            return
        raise ApprovalRequired(
            f"Action '{action.kind}' for {action.amount} requires human approval."
        )

"""Shared domain models for the orchestrator."""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class AccountType(str, Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT = "credit"
    MORTGAGE = "mortgage"
    UTILITY = "utility"


class Account(BaseModel):
    id: str
    name: str
    type: AccountType
    balance: float
    # The single checking account designated as discretionary "petty cash".
    is_petty_cash: bool = False
    # Annual interest rate as a percentage (e.g. 19.99 for a 19.99% APR card).
    # None when unknown; sourced from parsed statements / credit reports.
    apr: float | None = None


class Transaction(BaseModel):
    id: str
    account_id: str
    date: date
    amount: float  # negative = outflow
    description: str
    category: str | None = None


class GoalKind(str, Enum):
    # Grow a balance toward a target (optionally in a specific savings account).
    SAVINGS = "savings"
    # Pay a debt down to zero; progress is read from live account balances/trend.
    DEBT_PAYOFF = "debt_payoff"
    # A dated event funded through discrete milestones (e.g. a trip).
    MILESTONE = "milestone"


class Milestone(BaseModel):
    """A discrete, separately-timed piece of a milestone goal.

    ``payment_timing`` captures when the money is actually needed:
      * ``upfront``     — paid in full on/by ``due_date`` (e.g. airfare).
      * ``at_checkout`` — not charged until later (e.g. hotels at check-out),
                          so its savings timeline can run closer to ``due_date``.
    """

    name: str
    amount: float
    due_date: date | None = None
    payment_timing: str = "upfront"  # "upfront" | "at_checkout"
    # Amount set aside so far (manual; may be recomputed from a linked account).
    funded_amount: float = 0.0


class Goal(BaseModel):
    id: str
    name: str
    kind: GoalKind = GoalKind.SAVINGS
    # Optional so a goal can be as simple (name only) or rich as the user wants.
    target_amount: float | None = None
    target_date: date | None = None
    monthly_contribution: float | None = None
    # Name (or id) of a savings account whose balance tracks this goal's progress.
    linked_account: str | None = None
    # For debt payoff: the account names/ids being paid down (empty = all credit).
    target_accounts: list[str] = Field(default_factory=list)
    milestones: list[Milestone] = Field(default_factory=list)
    notes: str | None = None


class BudgetLine(BaseModel):
    category: str
    allocated: float
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return self.allocated - self.spent


class BudgetPlan(BaseModel):
    period: str  # e.g. "2026-07"
    monthly_income: float = 0.0
    lines: list[BudgetLine] = Field(default_factory=list)
    petty_cash_allocation: float = 0.0
    goal_contributions: dict[str, float] = Field(default_factory=dict)
    unallocated: float = 0.0
    goals: list[Goal] = Field(default_factory=list)

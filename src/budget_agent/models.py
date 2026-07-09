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


class Transaction(BaseModel):
    id: str
    account_id: str
    date: date
    amount: float  # negative = outflow
    description: str
    category: str | None = None


class Goal(BaseModel):
    id: str
    name: str
    target_amount: float
    target_date: date | None = None
    monthly_contribution: float | None = None


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

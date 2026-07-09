"""Append-only audit log for money-moving decisions.

Every time the approval gate evaluates a money-moving action, the outcome is recorded
here so there is a tamper-evident trail of what was auto-approved, human-approved, or
denied. The log is append-only: entries are never mutated or removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .approval import MoneyAction

_DECISIONS = frozenset({"auto_approved", "human_approved", "denied"})


@dataclass(frozen=True)
class AuditEntry:
    timestamp: datetime
    kind: str
    amount: float
    source_account_id: str
    dest_account_id: str
    reason: str
    decision: str


class AuditLog:
    """In-memory append-only audit log."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, action: MoneyAction, decision: str) -> AuditEntry:
        if decision not in _DECISIONS:
            raise ValueError(f"Unknown decision '{decision}'. Expected one of {sorted(_DECISIONS)}.")
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            kind=action.kind,
            amount=action.amount,
            source_account_id=action.source_account_id,
            dest_account_id=action.dest_account_id,
            reason=action.reason,
            decision=decision,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

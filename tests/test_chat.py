"""Tests for the /chat endpoint's snapshot handling: lookback window parsing
and the data_status signal that distinguishes a fetch failure from empty data.
"""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from budget_agent import service


class _FakeReasoner:
    """Records the analysis snapshot it was handed and echoes a canned reply."""

    def __init__(self) -> None:
        self.seen_analysis: dict[str, Any] | None = None

    def chat_and_plan(self, message, analysis, history, current_goals):
        self.seen_analysis = analysis
        return {
            "reply": "ok",
            "goals_updated": False,
            "goals": current_goals,
            "category_rules_proposed": [],
        }


class _FakeOrchestrator:
    def __init__(self, snapshot=None, error: Exception | None = None) -> None:
        self._snapshot = snapshot if snapshot is not None else {"accounts": []}
        self._error = error
        self.seen_days: int | None = None

    def snapshot(self, days: int = 30) -> dict[str, Any]:
        self.seen_days = days
        if self._error is not None:
            raise self._error
        return dict(self._snapshot)


def _wire(monkeypatch, reasoner, orch) -> TestClient:
    monkeypatch.setattr(service, "build_reasoner", lambda _s: reasoner)
    monkeypatch.setattr(service, "_orchestrator", lambda: orch)
    return TestClient(service.app)


def test_chat_uses_default_30_day_window(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(snapshot={"accounts": [], "total_inflow": 0.0})
    client = _wire(monkeypatch, reasoner, orch)

    resp = client.post("/chat", json={"message": "where can I save?"})

    assert resp.status_code == 200
    assert orch.seen_days == 30
    status = reasoner.seen_analysis["data_status"]
    assert status == {"ok": True, "lookback_days": 30}
    assert reasoner.seen_analysis["lookback_days"] == 30


def test_chat_widens_window_from_message(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(snapshot={"accounts": [], "total_inflow": 0.0})
    client = _wire(monkeypatch, reasoner, orch)

    client.post("/chat", json={"message": "how did I do over the past 6 months?"})

    assert orch.seen_days == 180
    assert reasoner.seen_analysis["data_status"]["lookback_days"] == 180


def test_chat_surfaces_degraded_data_status_on_snapshot_failure(monkeypatch):
    reasoner = _FakeReasoner()
    orch = _FakeOrchestrator(error=RuntimeError("aggregator down"))
    client = _wire(monkeypatch, reasoner, orch)

    resp = client.post("/chat", json={"message": "looking back 60 days"})

    assert resp.status_code == 200
    status = reasoner.seen_analysis["data_status"]
    assert status["ok"] is False
    assert status["lookback_days"] == 60
    assert "aggregator down" in status["error"]
    # A failed snapshot must not masquerade as real (empty) account data.
    assert "accounts" not in reasoner.seen_analysis

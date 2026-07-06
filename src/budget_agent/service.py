"""HTTP surface for the BudgetAgent orchestrator.

Exposes a health check and the read-only orchestration phases (analyze, plan). The
money-moving phases (propose/execute) are deliberately NOT exposed over HTTP while the
human-approval gate and execution adapter are still being built — see approval.py.

The tool clients (aggregator/analyzer/planner) are still stubs, so the read endpoints
return HTTP 501 until those are wired up (M4/M5). Health and info work today so the
container is live and its configuration is observable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .approval import ApprovalPolicy
from .config import Settings
from .models import Goal
from .orchestrator import Orchestrator
from .tools import AggregatorClient, AnalyzerClient, PlannerClient

app = FastAPI(title="budget-agent")


@lru_cache
def _settings() -> Settings:
    return Settings.from_env()


def _orchestrator() -> Orchestrator:
    s = _settings()
    return Orchestrator(
        aggregator=AggregatorClient(s.aggregator_url),
        analyzer=AnalyzerClient(s.analyzer_url),
        planner=PlannerClient(s.planner_url),
        policy=ApprovalPolicy(s.require_approval, s.auto_topup_cap),
    )


def _guard(fn: Callable[[], Any]) -> Any:
    """Run an orchestrator call, surfacing not-yet-implemented tools as HTTP 501."""
    try:
        return fn()
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail="Tool clients are not implemented yet (M4/M5).",
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def info() -> dict[str, Any]:
    s = _settings()
    return {
        "service": "budget-agent",
        "require_approval": s.require_approval,
        "phases": ["analyze", "plan", "propose", "approve", "execute", "track"],
        "tools": {
            "aggregator": s.aggregator_url,
            "analyzer": s.analyzer_url,
            "planner": s.planner_url,
        },
    }


@app.post("/analyze")
def analyze() -> Any:
    return _guard(lambda: _orchestrator().analyze())


class PlanRequest(BaseModel):
    analysis: dict[str, Any]
    goals: list[Goal] = []


@app.post("/plan")
def plan(req: PlanRequest) -> Any:
    return _guard(lambda: _orchestrator().plan(req.analysis, req.goals))

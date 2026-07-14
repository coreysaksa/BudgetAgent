"""HTTP surface for the BudgetAgent orchestrator.

Exposes read-only endpoints (analyze, plan, advise, recommend) plus an explicit,
guardrailed approval workflow (execute) that runs in **dry-run only** — actions are
validated against the approval gate and per-action limits, but no money is moved while
live money-movement integration remains deferred (high risk). See approval.py.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .approval import ApprovalPolicy, MoneyAction
from .config import Settings
from .models import BudgetPlan, Goal
from .notifications import Notifier
from .orchestrator import Orchestrator
from .reasoning import build_reasoner
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
        policy=ApprovalPolicy(
            require_approval=s.require_approval,
            auto_topup_cap=s.auto_topup_cap,
            max_action_amount=s.max_action_amount,
        ),
        notifier=Notifier(s.notification_webhook_url),
    )


def _guard(fn: Callable[[], Any]) -> Any:
    """Run an orchestrator call, surfacing tool/transport failures as HTTP errors."""
    try:
        return fn()
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail="This capability is not implemented yet (M5).",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream tool returned {exc.response.status_code}.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach an upstream tool: {exc}",
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


class AdviseRequest(BaseModel):
    analysis: dict[str, Any]
    plan: BudgetPlan


@app.post("/advise")
def advise(req: AdviseRequest) -> dict[str, str]:
    """Return an LLM narrative + recommendations for a plan (read-only, no execution)."""
    reasoner = build_reasoner(_settings())
    if reasoner is None:
        raise HTTPException(
            status_code=503,
            detail="Azure OpenAI is not configured (set AZURE_OPENAI_ENDPOINT).",
        )
    text = _guard(lambda: reasoner.advise(req.analysis, req.plan))
    return {"advice": text}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatGoal(BaseModel):
    name: str
    target_amount: float = 0.0
    monthly_contribution: float | None = None


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    goals: list[ChatGoal] = []


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """Conversational finance chat that can also build a plan and manage the
    user's savings goals (read-only w.r.t. money — never moves funds).

    Pulls a fresh spending analysis to ground the reply. If the tool services are
    unreachable (e.g. no bank linked yet), the chat still works with an empty
    snapshot. Returns ``{reply, goals_updated, goals}``; when ``goals_updated`` is
    true the caller should persist the returned goal set.
    """
    reasoner = build_reasoner(_settings())
    if reasoner is None:
        raise HTTPException(
            status_code=503,
            detail="Azure OpenAI is not configured (set AZURE_OPENAI_ENDPOINT).",
        )
    try:
        analysis = _orchestrator().analyze()
    except Exception:  # noqa: BLE001 - chat degrades gracefully without a snapshot
        analysis = {}
    history = [{"role": m.role, "content": m.content} for m in req.history]
    current_goals = [g.model_dump() for g in req.goals]
    return _guard(
        lambda: reasoner.chat_and_plan(req.message, analysis, history, current_goals)
    )


class RecommendRequest(BaseModel):
    goals: list[Goal] = []
    source_account_id: str = ""
    petty_cash_account_id: str = ""
    include_advice: bool = False


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    """Read-only recommendation: analyze -> plan -> propose. Never moves money."""
    rec = _guard(
        lambda: _orchestrator().recommend(
            req.goals, req.source_account_id, req.petty_cash_account_id
        )
    )
    result: dict[str, Any] = {
        "analysis": rec.analysis,
        "plan": rec.plan,
        "proposed_actions": rec.proposed_actions,
    }
    if req.include_advice:
        reasoner = build_reasoner(_settings())
        if reasoner is not None:
            result["advice"] = _guard(
                lambda: reasoner.advise(rec.analysis, rec.plan)
            )
    return result


class ActionRequest(BaseModel):
    kind: str
    amount: float
    source_account_id: str = ""
    dest_account_id: str = ""
    reason: str = ""


class ExecuteRequest(BaseModel):
    actions: list[ActionRequest]
    approvals: dict[str, bool] = {}


@app.post("/execute")
def execute(req: ExecuteRequest) -> dict[str, Any]:
    """Guardrailed approval workflow (DRY-RUN only).

    Validates each action against the approval gate + per-action limit and reports the
    would-be outcome. No money is moved: live execution is deferred (see approval.py).
    """
    actions = [
        MoneyAction(
            kind=a.kind,
            amount=a.amount,
            source_account_id=a.source_account_id,
            dest_account_id=a.dest_account_id,
            reason=a.reason,
        )
        for a in req.actions
    ]
    results = _guard(
        lambda: _orchestrator().execute(actions, req.approvals, dry_run=True)
    )
    return {"dry_run": True, "results": results}

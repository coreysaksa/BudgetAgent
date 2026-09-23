"""HTTP surface for the BudgetAgent orchestrator.

Exposes read-only endpoints (analyze, plan, advise, recommend) plus an explicit,
guardrailed approval workflow (execute) that runs in **dry-run only** — actions are
validated against the approval gate and per-action limits, but no money is moved while
live money-movement integration remains deferred (high risk). See approval.py.
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from typing import Any, Callable

import httpx
from fastapi import FastAPI, HTTPException
from openai import APIError, APIStatusError, RateLimitError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

from .approval import ApprovalPolicy, MoneyAction
from .chat_workflow import (
    ChatCommand,
    ChatWorkflow,
)
from .config import Settings
from .lookback import MAX_LOOKBACK_DAYS
from .models import (
    BudgetPlan,
    Goal,
    NecessityOverride,
    PaycheckInput,
    Windfall,
)
from .notifications import Notifier
from .orchestrator import Orchestrator
from .payoff import payoff_from_snapshot
from .payoff_scenario import build_payoff_scenario, reconcile_budget_baseline
from .reasoning import build_reasoner
from .tools import AggregatorClient, AnalyzerClient, PlannerClient

app = FastAPI(title="budget-agent")

_log = logging.getLogger(__name__)


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


def _is_transient_upstream(exc: BaseException) -> bool:
    """True when ``exc`` is an expected, temporary upstream outage (HTTP 429/503).

    An aggregator that is briefly rate-limited by Plaid returns 429; that is a
    normal, self-healing condition, not a bug. We log it concisely (no stack
    trace) so it doesn't trip "stack traces in console logs" alerts, while still
    logging genuinely unexpected failures with a full traceback.
    """
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in (429, 503)
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
    except RateLimitError as exc:
        # The Azure OpenAI deployment is briefly over its token/request rate limit
        # (the SDK already retried with backoff). Surface a clear, retryable 429
        # instead of an opaque 500 so the UI can tell the user to try again.
        _log.warning("assistant rate-limited: %s", exc)
        raise HTTPException(
            status_code=429,
            detail="The assistant is busy right now — please try again in a few seconds.",
        ) from exc
    except APIStatusError as exc:
        _log.warning("assistant returned %s: %s", exc.status_code, exc)
        raise HTTPException(
            status_code=502,
            detail=f"The assistant service returned an error ({exc.status_code}).",
        ) from exc
    except APIError as exc:
        _log.warning("assistant transport error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Couldn't reach the assistant service — please try again.",
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


class ChatMilestone(BaseModel):
    name: str
    amount: float = 0.0
    due_date: str | None = None
    payment_timing: str = "upfront"
    funded_amount: float = 0.0


class ChatGoal(BaseModel):
    # Mirrors the persisted goal shape so rich fields survive a chat round-trip
    # (the model echoes the full goal set on every turn). Extra/unknown fields
    # are ignored; every field is optional so simple goals still validate.
    id: str | None = None
    name: str
    kind: str = "savings"
    target_amount: float | None = None
    target_date: str | None = None
    monthly_contribution: float | None = None
    priority: int = 3
    horizon: str = "mid"
    deadline_type: str = "soft"
    minimum_monthly: float | None = None
    current_amount: float = 0.0
    status: str = "active"
    linked_account: str | None = None
    target_accounts: list[str] = []
    starting_balances: dict[str, float] = {}
    milestones: list[ChatMilestone] = []
    notes: str | None = None


PageContextValue = StrictBool | StrictInt | StrictFloat | StrictStr | None


class PageContext(BaseModel):
    """Untrusted UI context used only to orient the conversational reasoner."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    page: str | None = Field(default=None, max_length=100)
    title: str | None = Field(default=None, max_length=160)
    route: str | None = Field(default=None, max_length=200)
    selected_month: str | None = None
    summary: dict[str, PageContextValue] = Field(default_factory=dict, max_length=20)

    @field_validator("selected_month")
    @classmethod
    def validate_selected_month(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        try:
            parsed = date.fromisoformat(f"{value}-01")
        except ValueError as exc:
            raise ValueError("selected_month must use YYYY-MM format") from exc
        if parsed.strftime("%Y-%m") != value:
            raise ValueError("selected_month must use YYYY-MM format")
        return value

    @field_validator("summary")
    @classmethod
    def sanitize_summary(
        cls, value: dict[str, PageContextValue]
    ) -> dict[str, PageContextValue]:
        sanitized: dict[str, PageContextValue] = {}
        for key, item in value.items():
            clean_key = key.strip()
            if not clean_key or len(clean_key) > 64:
                raise ValueError("page context summary keys must be 1-64 characters")
            sanitized[clean_key] = item.strip()[:500] if isinstance(item, str) else item
        return sanitized


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    goals: list[ChatGoal] = []
    windfalls: list[Windfall] = []
    extra_income: list[dict[str, Any]] = []
    budget_baseline: list[dict[str, Any]] = []
    checking_buffer: float = 250.0
    payoff_plan_active: bool = False
    page_context: PageContext | None = None


class ExtraIncomeScenarioInput(BaseModel):
    id: str | None = None
    name: str
    amount: float
    frequency: str = "one_time"
    first_date: str | None = None
    end_date: str | None = None
    dates: list[str] = []
    status: str = "estimated"
    debt_percent: float = 100.0
    allocation_target: str = "auto"


class BudgetBaselineItemInput(BaseModel):
    id: str
    name: str
    category: str
    kind: str = "fixed"
    monthly_amount: float = 0.0
    inferred_monthly_amount: float | None = None
    due_day: int | None = Field(default=None, ge=1, le=31)
    periodic_amount: float | None = None
    frequency_months: int | None = Field(default=None, ge=1)
    next_due_date: str | None = None
    reserved_balance: float = 0.0
    funding_account_id: str | None = None
    review_required: bool = False
    review_prompt: str | None = None
    source: str = "inferred"
    confidence: str = "low"
    active: bool = True


class PayoffScenarioRequest(BaseModel):
    extra_income: list[ExtraIncomeScenarioInput] = []
    spending_adjustments: dict[str, float] = {}
    spending_adjustment_reasons: dict[str, str] = {}
    budget_baseline: list[BudgetBaselineItemInput] = []
    debt_allocation_percent: float = 100.0
    monthly_debt_extra: float | None = None
    checking_buffer: float = 250.0
    goals: list[ChatGoal] = []
    use_ai_suggestions: bool = True
    validate_feasibility: bool = True


class BudgetBaselineRequest(BaseModel):
    budget_baseline: list[BudgetBaselineItemInput] = []


class MerchantCandidate(BaseModel):
    merchant: str | None = None
    pending_id: str | None = None


class AdjudicateRequest(BaseModel):
    merchant: str
    candidates: list[MerchantCandidate] = []


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
    history = [{"role": m.role, "content": m.content} for m in req.history]
    command = ChatCommand(
        message=req.message,
        history=history,
        goals=[goal.model_dump() for goal in req.goals],
        windfalls=req.windfalls,
        extra_income=req.extra_income,
        budget_baseline=req.budget_baseline,
        checking_buffer=req.checking_buffer,
        payoff_plan_active=req.payoff_plan_active,
        page_context=(
            req.page_context.model_dump(exclude_none=True)
            if req.page_context is not None
            else None
        ),
    )
    workflow = ChatWorkflow(
        _orchestrator(),
        reasoner,
        build_payoff_scenario,
        _is_transient_upstream,
    )
    return _guard(lambda: workflow.execute(command))


@app.post("/adjudicate-merchants")
def adjudicate_merchants(req: AdjudicateRequest) -> dict[str, Any]:
    """Judge which candidate merchant names are the same business as ``merchant``.

    Used to auto-resolve borderline fuzzy matches when a user recategorizes a
    merchant, so obvious misspellings/aliases don't need a manual confirmation.
    Returns ``{"decisions": [{"merchant", "same"}, ...]}``; when Azure OpenAI is
    not configured, returns an empty decision set so the caller falls back to
    asking the user.
    """
    reasoner = build_reasoner(_settings())
    if reasoner is None:
        return {"decisions": []}
    candidates = [c.model_dump() for c in req.candidates]
    return _guard(lambda: reasoner.adjudicate_merchants(req.merchant, candidates))


class PayoffRequest(BaseModel):
    # Optional total dollars/month for cards; defaults to the derived surplus.
    monthly_budget: float | None = None
    # Optional monthly essentials set-aside (food/gas/tolls) override; when null
    # it's auto-derived from recent spending.
    reserve: float | None = None
    goals: list[ChatGoal] = []


class CashFlowRequest(BaseModel):
    as_of: str | None = None
    month: str | None = None
    windfalls: list[Windfall] = []
    paychecks: list[PaycheckInput] = []
    necessity_overrides: list[NecessityOverride] = []
    checking_buffer: float = 250.0


@app.post("/payoff-scenario")
def payoff_scenario(req: PayoffScenarioRequest) -> dict[str, Any]:
    """Build an editable, deterministic credit-card payoff what-if proposal."""
    orchestrator = _orchestrator()
    try:
        # Fetch the widest cached window first. Daily startup refresh and the
        # explicit refresh button control when Plaid is contacted.
        utility_history = orchestrator.snapshot(days=MAX_LOOKBACK_DAYS)
    except Exception as exc:  # noqa: BLE001 - current payoff analysis remains usable
        _log.warning("utility history unavailable for payoff scenario: %s", exc)
        utility_history = None
    analysis = _guard(lambda: orchestrator.snapshot(days=180))
    baseline = reconcile_budget_baseline(
        analysis,
        [item.model_dump(mode="json") for item in req.budget_baseline],
    )
    cash_flow = _guard(
        lambda: orchestrator.cash_flow_plan(
            analysis,
            [],
            checking_buffer=req.checking_buffer,
            budget_baseline=baseline,
        )
    )
    scenario = _guard(
        lambda: build_payoff_scenario(
            analysis,
            cash_flow,
            [goal.model_dump(mode="json") for goal in req.goals],
            utility_history=utility_history,
            extra_income=(
                None
                if req.use_ai_suggestions
                else [item.model_dump(mode="json") for item in req.extra_income]
            ),
            spending_adjustments=req.spending_adjustments,
            spending_adjustment_reasons=req.spending_adjustment_reasons,
            budget_baseline=baseline,
            debt_allocation_percent=req.debt_allocation_percent,
            monthly_debt_extra=req.monthly_debt_extra,
            validate_feasibility=req.validate_feasibility,
        )
    )
    critical = [
        item
        for item in cash_flow.get("clarification_questions") or []
        if item.get("critical")
    ]
    return {
        "data_ok": True,
        "ready": (
            isinstance(scenario.get("portfolio_plan"), dict)
            and not critical
            and (
                scenario.get("feasibility", {}).get("status") == "feasible"
                or scenario.get("underwater_approval", {}).get("eligible") is True
            )
        ),
        "scenario": scenario,
    }


@app.post("/budget-baseline")
def budget_baseline(req: BudgetBaselineRequest) -> dict[str, Any]:
    """Return current mandatory values while retaining confirmed periodic schedules."""
    analysis = _guard(lambda: _orchestrator().snapshot(days=180))
    return {
        "budget_baseline": reconcile_budget_baseline(
            analysis,
            [item.model_dump(mode="json") for item in req.budget_baseline],
        )
    }


@app.post("/cash-flow-plan")
def cash_flow_plan(req: CashFlowRequest) -> dict[str, Any]:
    """Deterministic paycheck survival targets and safe extra card capacity."""
    analysis = _guard(lambda: _orchestrator().snapshot(days=180))
    plan = _guard(
        lambda: _orchestrator().cash_flow_plan(
            analysis,
            req.windfalls,
            as_of=req.as_of,
            month=req.month,
            checking_buffer=req.checking_buffer,
            paychecks=req.paychecks,
            necessity_overrides=req.necessity_overrides,
        )
    )
    return {"data_ok": True, "plan": plan}


@app.post("/payoff")
def payoff(req: PayoffRequest) -> dict[str, Any]:
    """Deterministic month-by-month credit-card payoff schedule.

    Uses live account balances/APRs/promos plus the user's ``debt_payoff`` goals
    (per-card target dates and milestones). Read-only — never moves money.

    Returns ``configured`` (whether the user has set up a ``debt_payoff`` goal —
    the dashboard only shows the plan when they have), ``data_ok`` (whether the
    account snapshot loaded — so the UI can distinguish "still loading" from
    "no debt"), ``has_debt``, and the ``plan``.
    """
    goals = [g.model_dump() for g in req.goals]
    configured = any(str(g.get("kind")) == "debt_payoff" for g in goals)
    try:
        analysis = _orchestrator().snapshot()
        data_ok = True
    except Exception as exc:  # noqa: BLE001
        if _is_transient_upstream(exc):
            _log.warning("payoff snapshot unavailable (upstream busy): %s", exc)
        else:
            _log.warning("payoff snapshot failed: %s", exc, exc_info=True)
        analysis = {}
        data_ok = False

    if not data_ok:
        # Don't fabricate a "no debt" answer from an empty snapshot.
        return {
            "configured": configured,
            "data_ok": False,
            "has_debt": False,
            "plan": None,
        }

    plan = _guard(
        lambda: payoff_from_snapshot(
            analysis, goals, req.monthly_budget, reserve=req.reserve
        )
    )
    if plan is None:
        return {
            "configured": configured,
            "data_ok": True,
            "has_debt": False,
            "plan": None,
        }
    return {"configured": configured, "data_ok": True, "has_debt": True, "plan": plan}


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

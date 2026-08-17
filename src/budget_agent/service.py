"""HTTP surface for the BudgetAgent orchestrator.

Exposes read-only endpoints (analyze, plan, advise, recommend) plus an explicit,
guardrailed approval workflow (execute) that runs in **dry-run only** — actions are
validated against the approval gate and per-action limits, but no money is moved while
live money-movement integration remains deferred (high risk). See approval.py.
"""
from __future__ import annotations

import logging
from copy import deepcopy
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
from .config import Settings
from .lookback import MAX_LOOKBACK_DAYS, resolve_lookback_days
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
from .payoff_scenario import build_payoff_scenario
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
    status: str = "active"
    linked_account: str | None = None
    target_accounts: list[str] = []
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


class PayoffScenarioRequest(BaseModel):
    extra_income: list[ExtraIncomeScenarioInput] = []
    spending_adjustments: dict[str, float] = {}
    debt_allocation_percent: float = 100.0
    monthly_debt_extra: float | None = None
    checking_buffer: float = 250.0
    goals: list[ChatGoal] = []
    use_ai_suggestions: bool = True


class MerchantCandidate(BaseModel):
    merchant: str | None = None
    pending_id: str | None = None


class AdjudicateRequest(BaseModel):
    merchant: str
    candidates: list[MerchantCandidate] = []


def _trim_spending_tree(analysis: dict[str, Any], max_txns_per_sub: int = 8) -> None:
    """Cap the per-subcategory transaction lists so the chat prompt stays bounded.

    The analyzer embeds every transaction in ``spending_tree`` for the drill-down
    UI; the chat only needs the bucket/category/subcategory totals plus a few
    example transactions to spot savings, so keep the largest handful per leaf.
    """
    tree = analysis.get("spending_tree")
    if not isinstance(tree, list):
        return
    for bucket in tree:
        for category in bucket.get("categories", []):
            for sub in category.get("subcategories", []):
                txns = sub.get("transactions") or []
                if len(txns) > max_txns_per_sub:
                    sub["transactions"] = txns[:max_txns_per_sub]
                    sub["transactions_truncated"] = len(txns)


def _selected_month_lookback(selected_month: str | None) -> int:
    if not selected_month:
        return 0
    month_start = date.fromisoformat(f"{selected_month}-01")
    today = date.today()
    if month_start > today:
        return 0
    return min(MAX_LOOKBACK_DAYS, (today - month_start).days + 1)


def _has_cash_flow_context(message: str, history: list[dict[str, str]]) -> bool:
    text = " ".join(
        [turn.get("content", "") for turn in history if turn.get("role") == "user"]
        + [message]
    ).lower()
    return any(
        term in text
        for term in (
            "credit card",
            "payoff",
            "paycheck",
            "pay day",
            "paid on",
            "bonus",
            "windfall",
            "allowance",
            "security clearance",
            "sca",
            "mandatory",
            "discretionary",
            "survive",
        )
    )


def _requests_payoff_plan(message: str) -> bool:
    user_text = message.lower()
    return any(
        phrase in user_text
        for phrase in (
            "credit card payoff plan",
            "credit-card payoff plan",
            "credit card pay off plan",
            "payoff plan for my credit card",
            "payoff plan for my cards",
            "pay off plan for my credit card",
            "pay off plan for my cards",
            "plan to pay off my credit card",
            "plan to pay off my cards",
            "help me pay off my credit card",
            "help me pay off my cards",
            "recalculate my credit card payoff",
            "update my credit card payoff",
        )
    )


def _scenario_safe_extra(plan: dict[str, Any] | None) -> float:
    scenarios = (plan or {}).get("scenarios") or []
    if not scenarios:
        return 0.0
    return max(0.0, float(scenarios[0].get("safe_extra_payment") or 0.0))


def _merge_windfalls(
    requested: list[Windfall], extracted: list[Windfall]
) -> list[Windfall]:
    merged: list[Windfall] = []
    seen: set[tuple[str, float, str]] = set()
    for item in [*requested, *extracted]:
        key = (item.name.strip().lower(), round(item.amount, 2), item.date.isoformat())
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _suppress_draft_debt_goal_changes(
    result: dict[str, Any], current_goals: list[dict[str, Any]]
) -> None:
    if not result.get("goals_updated") or not isinstance(result.get("goals"), list):
        return
    current_debt = {
        str(goal.get("id") or goal.get("name") or "").strip().lower(): goal
        for goal in current_goals
        if goal.get("kind") == "debt_payoff"
    }
    filtered: list[dict[str, Any]] = []
    seen_debt: set[str] = set()
    for goal in result["goals"]:
        if not isinstance(goal, dict):
            continue
        if goal.get("kind") != "debt_payoff":
            filtered.append(goal)
            continue
        key = str(goal.get("id") or goal.get("name") or "").strip().lower()
        prior = current_debt.get(key)
        if prior is not None:
            filtered.append(prior)
            seen_debt.add(key)
    filtered.extend(goal for key, goal in current_debt.items() if key not in seen_debt)
    result["goals"] = filtered
    result["goals_updated"] = filtered != current_goals


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
    # Let the user widen the window conversationally ("looking back 60 days",
    # "past 6 months", "last quarter"); default to 30 days when they don't ask.
    # The window is sticky across turns: a follow-up that doesn't restate the
    # window keeps the last one the user named instead of snapping back to 30.
    conversational_lookback = resolve_lookback_days(req.message, req.history)
    selected_month = req.page_context.selected_month if req.page_context else None
    lookback_days = max(
        conversational_lookback,
        _selected_month_lookback(selected_month),
    )
    route = req.page_context.route if req.page_context else None
    scoped_month = (
        selected_month
        if selected_month
        and route
        and (
            route.startswith("/app/overview")
            or route.startswith("/app/transactions")
        )
        else None
    )
    try:
        if scoped_month:
            analysis = _orchestrator().snapshot(
                days=lookback_days,
                month=scoped_month,
            )
        else:
            analysis = _orchestrator().snapshot(days=lookback_days)
        data_status: dict[str, Any] = {"ok": True, "lookback_days": lookback_days}
    except Exception as exc:  # noqa: BLE001 - chat degrades gracefully without a snapshot
        # Don't swallow this silently: an aggregator/analyzer outage would
        # otherwise look identical to "no accounts connected" to the model. But a
        # transient upstream rate-limit (429) is expected and self-healing, so log
        # it without a stack trace to avoid tripping error-log alerts.
        if _is_transient_upstream(exc):
            _log.warning(
                "chat snapshot unavailable (lookback=%sd, upstream busy): %s",
                lookback_days,
                exc,
            )
        else:
            _log.warning(
                "chat snapshot failed (lookback=%sd): %s",
                lookback_days,
                exc,
                exc_info=True,
            )
        analysis = {}
        data_status = {
            "ok": False,
            "lookback_days": lookback_days,
            "error": f"{type(exc).__name__}: {exc}",
        }
    planner_analysis = deepcopy(analysis) if analysis else {}
    if analysis:
        _trim_spending_tree(analysis)
        analysis["lookback_days"] = lookback_days
    history = [{"role": m.role, "content": m.content} for m in req.history]
    current_goals = [g.model_dump() for g in req.goals]
    extracted: dict[str, Any] = {
        "windfalls": [],
        "paychecks": [],
        "necessity_overrides": [],
        "clarifications": [],
    }
    payoff_context = req.payoff_plan_active or _requests_payoff_plan(req.message)
    if analysis and payoff_context:
        try:
            extracted = reasoner.extract_cash_flow_inputs(req.message, history)
        except Exception as exc:  # noqa: BLE001 - base chat remains available
            _log.warning("cash-flow input extraction unavailable: %s", exc)
    payoff_plan: dict[str, Any] | None = None
    payoff_scenario: dict[str, Any] | None = None
    cash_flow_plan: dict[str, Any] | None = None
    payoff_ready = False
    if analysis and payoff_context:
        try:
            try:
                utility_history = _orchestrator().snapshot(days=730)
            except Exception as exc:  # noqa: BLE001 - payoff chat can use fallback reserve
                _log.warning("utility history unavailable for payoff chat: %s", exc)
                utility_history = None
            structured_windfalls = _merge_windfalls(
                req.windfalls,
                [Windfall.model_validate(item) for item in extracted["windfalls"]],
            )
            cash_flow_plan = _orchestrator().cash_flow_plan(
                planner_analysis,
                structured_windfalls,
                checking_buffer=req.checking_buffer,
                paychecks=[
                    PaycheckInput.model_validate(item) for item in extracted["paychecks"]
                ],
                necessity_overrides=[
                    NecessityOverride.model_validate(item)
                    for item in extracted["necessity_overrides"]
                ],
            )
            baseline_cash_flow = _orchestrator().cash_flow_plan(
                planner_analysis,
                [],
                checking_buffer=req.checking_buffer,
                paychecks=[
                    PaycheckInput.model_validate(item) for item in extracted["paychecks"]
                ],
                necessity_overrides=[
                    NecessityOverride.model_validate(item)
                    for item in extracted["necessity_overrides"]
                ],
            )
            cash_flow_plan.setdefault("clarification_questions", []).extend(
                {
                    "code": "missing-conversation-input",
                    "question": question,
                    "context": None,
                    "critical": True,
                }
                for question in extracted["clarifications"]
            )
            analysis["cash_flow_plan"] = cash_flow_plan
            minimum_total = sum(
                max(0.0, float(account.get("minimum_payment") or 0.0))
                for account in planner_analysis.get("accounts") or []
                if account.get("type") == "credit"
            )
            baseline_extra = max(
                0.0,
                float(
                    baseline_cash_flow.get("recurring_safe_extra_payment") or 0.0
                ),
            )
            current_baseline_extra = _scenario_safe_extra(baseline_cash_flow)
            confirmed_extra = _scenario_safe_extra(cash_flow_plan)
            confirmed_windfall_total = sum(
                item.amount
                for item in structured_windfalls
                if item.status == "confirmed"
            )
            safe_windfall_extra = max(
                0.0, confirmed_extra - current_baseline_extra
            )
            confirmed_debt_percent = (
                min(100.0, safe_windfall_extra * 100.0 / confirmed_windfall_total)
                if confirmed_windfall_total > 0
                else 100.0
            )
            estimated_windfall_total = sum(
                item.amount
                for item in structured_windfalls
                if item.status == "estimated"
            )
            scenarios = cash_flow_plan.get("scenarios") or []
            estimated_scenario_extra = (
                max(
                    0.0,
                    float(scenarios[1].get("safe_extra_payment") or 0.0)
                    - confirmed_extra,
                )
                if len(scenarios) > 1
                else 0.0
            )
            estimated_debt_percent = (
                min(
                    100.0,
                    estimated_scenario_extra * 100.0 / estimated_windfall_total,
                )
                if estimated_windfall_total > 0
                else 100.0
            )
            payoff_scenario = build_payoff_scenario(
                planner_analysis,
                baseline_cash_flow,
                [],
                utility_history=utility_history,
                extra_income=[
                    {
                        "name": item.name,
                        "amount": item.amount,
                        "frequency": "one_time",
                        "first_date": item.date.isoformat(),
                        "status": item.status,
                        "debt_percent": (
                            confirmed_debt_percent
                            if item.status == "confirmed"
                            else estimated_debt_percent
                        ),
                    }
                    for item in structured_windfalls
                ],
            )
            payoff_plan = payoff_scenario.get("plan")
            critical_questions = [
                item
                for item in cash_flow_plan.get("clarification_questions") or []
                if item.get("critical")
            ]
            payoff_ready = payoff_plan is not None and not critical_questions
            payoff_ready = payoff_ready and (
                payoff_scenario.get("feasibility", {}).get("status") == "feasible"
            )
            if payoff_plan is not None:
                payoff_plan["minimum_payment_total"] = round(minimum_total, 2)
                payoff_plan["safe_extra_payment"] = round(baseline_extra, 2)
                payoff_plan["initial_extra_payment"] = round(
                    safe_windfall_extra, 2
                )
                prompt_plan = payoff_plan
                sched = payoff_plan.get("schedule") or []
                if len(sched) > 24:
                    prompt_plan = {
                        **payoff_plan,
                        "schedule": sched[:24],
                        "schedule_truncated": True,
                    }
                analysis["debt_payoff_plan"] = prompt_plan
        except Exception as exc:  # noqa: BLE001 - chat remains usable without this plan
            _log.warning("cash-flow plan unavailable for chat: %s", exc)
    # Always surface how the data load went so the model can distinguish a
    # temporary fetch failure from a genuinely empty account set.
    analysis = analysis or {}
    if req.page_context is not None:
        analysis["page_context"] = req.page_context.model_dump(exclude_none=True)
    analysis["data_status"] = data_status
    result = _guard(
        lambda: reasoner.chat_and_plan(req.message, analysis, history, current_goals)
    )
    if payoff_context:
        _suppress_draft_debt_goal_changes(result, current_goals)
    result.update(
        {
            "payoff_plan_status": "draft" if payoff_plan is not None else "none",
            "payoff_plan_ready": payoff_ready,
            "payoff_plan": payoff_plan,
            "cash_flow_plan": cash_flow_plan,
            "payoff_scenario": payoff_scenario,
        }
    )
    return result


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
        # Fetch the widest window first. The aggregator then serves the 180-day
        # snapshot from that cache instead of making two sequential Plaid pulls.
        utility_history = orchestrator.snapshot(days=MAX_LOOKBACK_DAYS)
    except Exception as exc:  # noqa: BLE001 - current payoff analysis remains usable
        _log.warning("utility history unavailable for payoff scenario: %s", exc)
        utility_history = None
    analysis = _guard(lambda: orchestrator.snapshot(days=180))
    cash_flow = _guard(
        lambda: orchestrator.cash_flow_plan(
            analysis,
            [],
            checking_buffer=req.checking_buffer,
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
            debt_allocation_percent=req.debt_allocation_percent,
            monthly_debt_extra=req.monthly_debt_extra,
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
            and scenario.get("feasibility", {}).get("status") == "feasible"
        ),
        "scenario": scenario,
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

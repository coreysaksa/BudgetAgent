"""Application workflow for a conversational BudgetAI turn."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from .lookback import MAX_LOOKBACK_DAYS, resolve_lookback_days
from .models import NecessityOverride, PaycheckInput, Windfall
from .orchestrator import Orchestrator
from .payoff_scenario import reconcile_budget_baseline

_log = logging.getLogger(__name__)


class ChatReasoner(Protocol):
    def extract_cash_flow_inputs(
        self, message: str, history: list[dict[str, str]]
    ) -> dict[str, Any]: ...

    def chat_and_plan(
        self,
        message: str,
        analysis: dict[str, Any],
        history: list[dict[str, str]],
        current_goals: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ChatCommand:
    message: str
    history: list[dict[str, str]]
    goals: list[dict[str, Any]]
    windfalls: list[Windfall]
    extra_income: list[dict[str, Any]]
    budget_baseline: list[dict[str, Any]]
    checking_buffer: float
    payoff_plan_active: bool
    page_context: dict[str, Any] | None


@dataclass(frozen=True)
class PayoffChatContext:
    plan: dict[str, Any] | None = None
    scenario: dict[str, Any] | None = None
    cash_flow_plan: dict[str, Any] | None = None
    ready: bool = False


def trim_spending_tree(
    analysis: dict[str, Any], max_txns_per_sub: int = 8
) -> None:
    """Cap transaction samples included in the conversational prompt."""
    tree = analysis.get("spending_tree")
    if not isinstance(tree, list):
        return
    for bucket in tree:
        for category in bucket.get("categories", []):
            for subcategory in category.get("subcategories", []):
                transactions = subcategory.get("transactions") or []
                if len(transactions) > max_txns_per_sub:
                    subcategory["transactions"] = transactions[:max_txns_per_sub]
                    subcategory["transactions_truncated"] = len(transactions)


def selected_month_lookback(
    selected_month: str | None,
    today: date | None = None,
) -> int:
    if not selected_month:
        return 0
    month_start = date.fromisoformat(f"{selected_month}-01")
    current_date = today or date.today()
    if month_start > current_date:
        return 0
    return min(MAX_LOOKBACK_DAYS, (current_date - month_start).days + 1)


def requests_payoff_plan(message: str) -> bool:
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


def merge_windfalls(
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


def suppress_draft_debt_goal_changes(
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


class ChatWorkflow:
    def __init__(
        self,
        orchestrator: Orchestrator,
        reasoner: ChatReasoner,
        scenario_builder: Callable[..., dict[str, Any]],
        is_transient_upstream: Callable[[BaseException], bool],
        today: Callable[[], date] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._reasoner = reasoner
        self._scenario_builder = scenario_builder
        self._is_transient_upstream = is_transient_upstream
        self._today = today or date.today

    def execute(self, command: ChatCommand) -> dict[str, Any]:
        current_date = self._today()
        lookback_days = max(
            resolve_lookback_days(command.message, command.history),
            selected_month_lookback(
                self._selected_month(command),
                current_date,
            ),
        )
        analysis, data_status = self._load_analysis(command, lookback_days)
        planner_analysis = deepcopy(analysis) if analysis else {}
        if analysis:
            trim_spending_tree(analysis)
            analysis["lookback_days"] = lookback_days

        payoff_context_requested = (
            command.payoff_plan_active or requests_payoff_plan(command.message)
        )
        payoff_context = self._build_payoff_context(
            command,
            analysis,
            planner_analysis,
            payoff_context_requested,
            current_date,
        )

        analysis = analysis or {}
        if command.page_context is not None:
            analysis["page_context"] = command.page_context
        analysis["data_status"] = data_status
        result = self._reasoner.chat_and_plan(
            command.message,
            analysis,
            command.history,
            command.goals,
        )
        if payoff_context_requested:
            suppress_draft_debt_goal_changes(result, command.goals)
        result.update(
            {
                "payoff_plan_status": (
                    "draft" if payoff_context.plan is not None else "none"
                ),
                "payoff_plan_ready": payoff_context.ready,
                "payoff_plan": payoff_context.plan,
                "cash_flow_plan": payoff_context.cash_flow_plan,
                "payoff_scenario": payoff_context.scenario,
            }
        )
        return result

    @staticmethod
    def _selected_month(command: ChatCommand) -> str | None:
        if command.page_context is None:
            return None
        value = command.page_context.get("selected_month")
        return str(value) if value else None

    def _load_analysis(
        self,
        command: ChatCommand,
        lookback_days: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        selected_month = self._selected_month(command)
        route = (
            str(command.page_context.get("route") or "")
            if command.page_context
            else ""
        )
        scoped_month = (
            selected_month
            if selected_month
            and (
                route.startswith("/app/overview")
                or route.startswith("/app/transactions")
            )
            else None
        )
        try:
            analysis = self._orchestrator.snapshot(
                days=lookback_days,
                **({"month": scoped_month} if scoped_month else {}),
            )
            return analysis, {"ok": True, "lookback_days": lookback_days}
        except Exception as exc:  # noqa: BLE001 - chat degrades without a snapshot
            if self._is_transient_upstream(exc):
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
            return {}, {
                "ok": False,
                "lookback_days": lookback_days,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _build_payoff_context(
        self,
        command: ChatCommand,
        analysis: dict[str, Any],
        planner_analysis: dict[str, Any],
        requested: bool,
        current_date: date,
    ) -> PayoffChatContext:
        if not analysis or not requested:
            return PayoffChatContext()

        extracted = self._extract_cash_flow_inputs(command)
        try:
            utility_history = self._load_utility_history()
            structured_windfalls = merge_windfalls(
                command.windfalls,
                [
                    Windfall.model_validate(item)
                    for item in extracted["windfalls"]
                ],
            )
            baseline = reconcile_budget_baseline(
                planner_analysis,
                command.budget_baseline,
            )
            paychecks = [
                PaycheckInput.model_validate(item) for item in extracted["paychecks"]
            ]
            necessity_overrides = [
                NecessityOverride.model_validate(item)
                for item in extracted["necessity_overrides"]
            ]
            cash_flow_plan = self._orchestrator.cash_flow_plan(
                planner_analysis,
                structured_windfalls,
                checking_buffer=command.checking_buffer,
                paychecks=paychecks,
                necessity_overrides=necessity_overrides,
                budget_baseline=baseline,
            )
            baseline_cash_flow = self._orchestrator.cash_flow_plan(
                planner_analysis,
                [],
                checking_buffer=command.checking_buffer,
                paychecks=paychecks,
                necessity_overrides=necessity_overrides,
                budget_baseline=baseline,
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
            scenario = self._scenario_builder(
                planner_analysis,
                baseline_cash_flow,
                [],
                utility_history=utility_history,
                extra_income=(
                    command.extra_income
                    or [
                        {
                            "name": item.name,
                            "amount": item.amount,
                            "frequency": "one_time",
                            "first_date": item.date.isoformat(),
                            "status": item.status,
                        }
                        for item in structured_windfalls
                    ]
                ),
                budget_baseline=baseline,
                validate_feasibility=False,
            )
            analysis["budget_baseline"] = baseline
            analysis["extra_income_allocations"] = scenario.get("extra_income", [])
            plan = scenario.get("plan")
            self._enrich_payoff_plan(
                plan,
                scenario,
                baseline_cash_flow,
                planner_analysis,
                current_date,
            )
            if isinstance(plan, dict):
                analysis["debt_payoff_plan"] = self._prompt_plan(plan)
            critical_questions = [
                item
                for item in cash_flow_plan.get("clarification_questions") or []
                if item.get("critical")
            ]
            ready = isinstance(plan, dict) and not critical_questions
            ready = ready and (
                scenario.get("feasibility", {}).get("status") == "feasible"
            )
            return PayoffChatContext(
                plan=plan if isinstance(plan, dict) else None,
                scenario=scenario,
                cash_flow_plan=cash_flow_plan,
                ready=ready,
            )
        except Exception as exc:  # noqa: BLE001 - base chat remains available
            _log.warning("cash-flow plan unavailable for chat: %s", exc)
            return PayoffChatContext()

    def _extract_cash_flow_inputs(
        self,
        command: ChatCommand,
    ) -> dict[str, Any]:
        empty: dict[str, Any] = {
            "windfalls": [],
            "paychecks": [],
            "necessity_overrides": [],
            "clarifications": [],
        }
        try:
            return self._reasoner.extract_cash_flow_inputs(
                command.message,
                command.history,
            )
        except Exception as exc:  # noqa: BLE001 - base chat remains available
            _log.warning("cash-flow input extraction unavailable: %s", exc)
            return empty

    def _load_utility_history(self) -> dict[str, Any] | None:
        try:
            return self._orchestrator.snapshot(days=MAX_LOOKBACK_DAYS)
        except Exception as exc:  # noqa: BLE001 - fallback reserve remains available
            _log.warning("utility history unavailable for payoff chat: %s", exc)
            return None

    def _enrich_payoff_plan(
        self,
        plan: Any,
        scenario: dict[str, Any],
        baseline_cash_flow: dict[str, Any],
        planner_analysis: dict[str, Any],
        current_date: date,
    ) -> None:
        if not isinstance(plan, dict):
            return
        minimum_total = sum(
            max(0.0, float(account.get("minimum_payment") or 0.0))
            for account in planner_analysis.get("accounts") or []
            if account.get("type") == "credit"
        )
        plan["minimum_payment_total"] = round(minimum_total, 2)
        plan["safe_extra_payment"] = round(
            max(
                0.0,
                float(
                    baseline_cash_flow.get("recurring_safe_extra_payment")
                    or 0.0
                ),
            ),
            2,
        )
        plan["initial_extra_payment"] = round(
            float(
                scenario.get("extra_payments_by_month", {}).get(
                    current_date.strftime("%Y-%m"),
                    0.0,
                )
            ),
            2,
        )

    @staticmethod
    def _prompt_plan(plan: dict[str, Any]) -> dict[str, Any]:
        schedule = plan.get("schedule") or []
        if len(schedule) <= 24:
            return plan
        return {
            **plan,
            "schedule": schedule[:24],
            "schedule_truncated": True,
        }

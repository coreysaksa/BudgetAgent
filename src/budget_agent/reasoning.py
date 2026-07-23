"""LLM reasoning layer: turns the numeric analysis/plan into human guidance.

The orchestrator's tools produce structured data (spending analysis, a budget plan,
proposed money actions). This layer asks an Azure OpenAI chat model to explain that
data and recommend next steps in plain language. It never moves money — the approval
gate in ``approval.py`` remains the only path to execution.

The model client is injected (``Protocol``) so tests run without a live endpoint, and
``build_reasoner`` constructs the real Azure OpenAI client only when configured.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

from .approval import MoneyAction
from .config import Settings
from .models import BudgetPlan

_SYSTEM_PROMPT = (
    "You are BudgetAI, a careful personal-finance assistant. Given a spending analysis, "
    "a proposed monthly budget, and any recommended money movements, explain the plan in "
    "plain language and give 3-5 concrete, prioritized recommendations. Be concise and "
    "specific with numbers. Never claim to have moved money; all transfers require the "
    "user's explicit approval."
)

_CHAT_SYSTEM_PROMPT = (
    "You are BudgetAI, a friendly, careful personal-finance assistant chatting with the "
    "user about their money. You may be given a JSON snapshot of their current spending "
    "analysis. The snapshot includes an `accounts` list where each account has a `name`, "
    "`type` (checking/savings/credit/mortgage/loan), `balance` (negative balances are "
    "amounts owed on credit cards, mortgages, and loans), and `apr` (its annual interest "
    "rate as a percentage, or null if unknown). Each account may also carry a `promos` "
    "list of promotional/introductory/balance-transfer offers, each with `promo_type`, "
    "`apr` (the promo rate, e.g. 0 for a 0% offer), `end_date` (when it expires and the "
    "balance reverts to the standard APR), and `balance` (the amount that rate applies "
    "to). It also has spending by category and "
    "income vs. outflow. Answer questions clearly and specifically, citing real numbers "
    "from the snapshot when relevant — reference actual balances and, when comparing debts "
    "or savings, their APRs (e.g. steer extra payments toward the highest-APR balance). "
    "When an account has promos, warn about expiring promotional rates and prioritise "
    "paying off a promo balance at or just before its `end_date` so it isn't hit by the "
    "much higher standard APR (a 0% promo balance is not urgent until its end_date nears, "
    "but must be cleared by then). "
    "The snapshot also includes a `spending_tree`: the user's outflow grouped into two "
    "buckets — `mandatory` (credit-score impacting debt plus essentials like housing, "
    "utilities, groceries, insurance) and `discretionary` (dining, entertainment, "
    "shopping, travel). Each bucket lists `categories`, each category its `subcategories`, "
    "and each subcategory its `transactions` (a sample; `transactions_truncated` gives the "
    "true count). Use this to analyze where money goes and recommend concrete savings — "
    "focus cuts on discretionary spend, name the specific categories/merchants driving it, "
    "and quantify the monthly saving; treat mandatory spending as harder to cut. "
    "If the snapshot includes a `debt_payoff_plan`, it is a pre-computed, authoritative "
    "month-by-month payoff schedule — use its exact payment amounts, payoff dates and interest "
    "rather than estimating your own, and relay any `warnings` when it isn't `feasible`. "
    "Highlight concrete, prioritized areas where they could improve their budget. Keep "
    "replies concise. If the snapshot is empty or has no accounts, say you don't see any "
    "connected accounts yet and suggest connecting a bank or uploading a statement. You can "
    "explain and advise, but you never move money — transfers require the user's explicit "
    "approval."
)

_PLAN_SYSTEM_PROMPT = (
    "You are BudgetAI, a friendly, careful personal-finance assistant. The user manages "
    "their goals by talking to you — this is the ONLY way their goals change.\n\n"
    "You are given (as JSON) the user's current goals, and possibly a snapshot of their "
    "spending analysis. The snapshot includes an `accounts` list (each with `id`, `name`, "
    "`type`, `balance`, and `apr` — the annual interest rate as a percentage, or null), plus "
    "spending by category and income vs. outflow. Use real balances and APRs when shaping a "
    "plan: fund goals from available savings/checking balances, account for how much "
    "high-APR debt is costing, and prefer paying down the highest-APR balances. Accounts "
    "may also carry a `promos` list (promotional/intro/balance-transfer offers, each with "
    "`promo_type`, `apr`, `end_date`, and `balance`); when planning a debt_payoff, schedule "
    "extra payments so each promotional balance is cleared at or before its `end_date`, "
    "before it reverts to the standard APR.\n\n"
    "SPENDING DETAIL: the snapshot includes a `spending_tree` grouping outflow into "
    "`mandatory` (credit-score impacting debt + essentials like housing, utilities, "
    "groceries, insurance) and `discretionary` (dining, entertainment, shopping, travel) "
    "buckets, each with categories -> subcategories -> sample `transactions`. Use it to "
    "ground savings recommendations in real spend and to protect mandatory spending when "
    "shaping a budget.\n\n"
    "AUTHORITATIVE PAYOFF SCHEDULE: the snapshot may include a `debt_payoff_plan` object — a "
    "pre-computed, deterministic month-by-month credit-card payoff schedule. When present, it "
    "is the source of truth: present ITS numbers exactly and DO NOT recompute or invent "
    "different payment amounts, payoff dates, or interest. It contains `monthly_budget` (total "
    "$/month across cards), `cards` (each with `name`, `starting_balance`, `apr`, `deadline`, "
    "`payoff_month`, `on_time`, `total_interest`), a `schedule` (per month: `month` plus "
    "per-card `payment`/`interest`/`remaining`), `total_interest`, `months_to_debt_free`, and "
    "`feasible`/`warnings`. When the user asks for a payoff plan, lay out the per-card payoff "
    "dates and a concise month-by-month 'pay $X to CardA, $Y to CardB' table from this data. If "
    "`feasible` is false, clearly relay each warning (e.g. a card that can't be cleared by its "
    "deadline) and suggest raising the monthly budget or moving the date.\n\n"
    "GOALS CAN BE SIMPLE OR RICH. Each goal is an object with these fields (include only "
    "what's relevant; omit or null the rest):\n"
    "  - name: string (required)\n"
    "  - kind: one of \"savings\" (grow a balance toward a target), \"debt_payoff\" (pay a "
    "debt down to zero), or \"milestone\" (a dated event funded through sub-milestones). "
    "Default \"savings\".\n"
    "  - target_amount: number or null — how much to save, or the debt size to pay off.\n"
    "  - target_date: \"YYYY-MM-DD\" or null — when they want it done by.\n"
    "  - monthly_contribution: number or null — planned monthly amount. If they give a "
    "target and date, estimate this; if they give a monthly amount, you may estimate the "
    "target.\n"
    "  - linked_account: string or null — for a savings goal, the NAME of the savings "
    "account that holds the money for this goal (match a `name` from the snapshot accounts, "
    "e.g. \"Travel Savings\"). Progress is then tracked from that account's real balance. "
    "Leave null if no dedicated account.\n"
    "  - target_accounts: array of account NAMES — for a debt_payoff goal, the card/loan "
    "accounts being paid down (match `name`s from the snapshot). Empty means all credit "
    "accounts. Debt goals are tracked from live balances, NOT a linked savings account.\n"
    "  - milestones: array of {\"name\", \"amount\", \"due_date\" (\"YYYY-MM-DD\" or null), "
    "\"payment_timing\" (\"upfront\" if paid in full on the date like airfare, or "
    "\"at_checkout\" if not charged until later like hotels)} — for milestone goals, break "
    "the goal into these dated pieces so their savings timelines can differ.\n"
    "  - notes: string or null — any special context worth remembering.\n\n"
    "Behave as follows:\n"
    "- When the user tells you about a goal to add, change, or remove, or asks you to plan "
    "for a goal: write a short, concrete plan in `reply` — reference real numbers from the "
    "snapshot, suggest a realistic monthly contribution and rough timeline, and for milestone "
    "goals note how each milestone's timing affects when the money is needed. Then set "
    "`goals_updated` to true and return in `goals` the COMPLETE updated list: keep every "
    "existing goal the user did NOT ask to change AND RETURN EACH OF ITS FIELDS VERBATIM "
    "(including id, linked_account, target_accounts, milestones), apply their additions and "
    "edits, and omit any goal they asked to remove.\n"
    "- When the user is just asking a question and NOT changing goals, set `goals_updated` "
    "to false and return the current goals unchanged in `goals`.\n"
    "- Never invent goals the user didn't ask for. Preserve each existing goal's `id`.\n\n"
    "Respond with ONLY a compact JSON object of the form: "
    "{\"reply\": string, \"goals_updated\": boolean, \"goals\": [goal, ...]}. "
    "Do not wrap it in markdown or add any text outside the JSON."
)


class ChatClient(Protocol):
    """Minimal structural type for an OpenAI-style chat client."""

    @property
    def chat(self) -> Any: ...


_MISSING = object()
_GOAL_KINDS = {"savings", "debt_payoff", "milestone"}
_PAYMENT_TIMINGS = {"upfront", "at_checkout"}


def _norm_name(name: Any) -> str:
    return " ".join(str(name or "").lower().split())


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_milestones(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for m in value:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        timing = str(m.get("payment_timing") or "upfront").strip().lower()
        if timing not in _PAYMENT_TIMINGS:
            timing = "upfront"
        out.append(
            {
                "name": name,
                "amount": _to_float(m.get("amount")) or 0.0,
                "due_date": _to_str(m.get("due_date")),
                "payment_timing": timing,
                "funded_amount": _to_float(m.get("funded_amount")) or 0.0,
            }
        )
    return out


def _merge_goal(g: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    """Normalize a model-returned goal, inheriting any field the model OMITTED
    (key absent) from the matching prior goal so unchanged rich data isn't lost.
    A field the model returns explicitly (even null/empty) is respected.
    """

    def pick(key: str, coerce, default):
        raw = g.get(key, _MISSING)
        if raw is _MISSING:
            return prior.get(key, default)
        return coerce(raw)

    kind = pick("kind", lambda v: str(v or "savings").strip().lower(), "savings")
    if kind not in _GOAL_KINDS:
        kind = "savings"

    return {
        "id": _to_str(g.get("id")) or _to_str(prior.get("id")),
        "name": str(g.get("name") or "").strip(),
        "kind": kind,
        "target_amount": pick("target_amount", _to_float, None),
        "target_date": pick("target_date", _to_str, None),
        "monthly_contribution": pick("monthly_contribution", _to_float, None),
        "linked_account": pick("linked_account", _to_str, None),
        "target_accounts": pick(
            "target_accounts",
            lambda v: [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, list)
            else [],
            [],
        ),
        "milestones": pick("milestones", _parse_milestones, []),
        "notes": pick("notes", _to_str, None),
    }


class Reasoner:
    def __init__(self, client: ChatClient, deployment: str) -> None:
        self._client = client
        self._deployment = deployment

    def advise(
        self,
        analysis: dict[str, Any],
        plan: BudgetPlan,
        actions: list[MoneyAction] | None = None,
    ) -> str:
        user_payload = {
            "analysis": analysis,
            "plan": plan.model_dump(mode="json"),
            "proposed_actions": [
                {
                    "kind": a.kind,
                    "amount": a.amount,
                    "reason": a.reason,
                }
                for a in (actions or [])
            ],
        }
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, default=str)},
            ],
        )
        return resp.choices[0].message.content or ""

    def chat(
        self,
        message: str,
        analysis: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Free-form conversational reply grounded in the user's spending snapshot.

        ``history`` is a list of prior ``{"role": "user"|"assistant", "content": ...}``
        turns; ``analysis`` is the latest spending analysis (may be empty when no
        accounts are connected yet).
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _CHAT_SYSTEM_PROMPT}
        ]
        if analysis:
            messages.append(
                {
                    "role": "system",
                    "content": "Current financial snapshot (JSON):\n"
                    + json.dumps(analysis, default=str),
                }
            )
        for turn in history or []:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    def chat_and_plan(
        self,
        message: str,
        analysis: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        current_goals: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Conversational reply that can also add/edit/remove savings goals.

        Returns ``{"reply": str, "goals_updated": bool, "goals": list}``. When
        ``goals_updated`` is true, ``goals`` is the complete desired goal set the
        caller should persist (replacing the prior set); otherwise ``goals`` is
        the unchanged current set and callers should leave storage untouched.
        """
        current_goals = current_goals or []
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Current goals (JSON):\n"
                + json.dumps(current_goals, default=str),
            },
        ]
        if analysis:
            messages.append(
                {
                    "role": "system",
                    "content": "Current financial snapshot (JSON):\n"
                    + json.dumps(analysis, default=str),
                }
            )
        for turn in history or []:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        return self._parse_plan_response(raw, current_goals)

    @staticmethod
    def _parse_plan_response(
        raw: str, current_goals: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Parse the model's JSON reply, degrading gracefully on malformed output."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Fall back to treating the whole text as a plain reply, changing nothing.
            return {"reply": raw, "goals_updated": False, "goals": current_goals}

        reply = str(data.get("reply") or "")
        goals_updated = bool(data.get("goals_updated"))
        goals_raw = data.get("goals")
        if not goals_updated or not isinstance(goals_raw, list):
            return {"reply": reply, "goals_updated": False, "goals": current_goals}

        # Index current goals by id and by normalized name so we can preserve ids
        # and inherit any rich fields the model omitted from an unchanged goal.
        by_id = {str(g.get("id")): g for g in current_goals if g.get("id")}
        by_name = {
            _norm_name(g.get("name")): g for g in current_goals if g.get("name")
        }

        goals: list[dict[str, Any]] = []
        for g in goals_raw:
            if not isinstance(g, dict):
                continue
            name = str(g.get("name") or "").strip()
            if not name:
                continue
            prior = by_id.get(str(g.get("id"))) or by_name.get(_norm_name(name)) or {}
            goals.append(_merge_goal(g, prior))
        return {"reply": reply, "goals_updated": True, "goals": goals}


def build_reasoner(settings: Settings) -> Reasoner | None:
    """Build a Reasoner backed by Azure OpenAI, or None if not configured.

    Uses Entra ID (managed identity / DefaultAzureCredential) — no API keys — matching
    the ``disableLocalAuth`` setting on the Azure OpenAI account.
    """
    if not settings.azure_openai_endpoint:
        return None

    # Imported lazily so unit tests (which inject a fake client) don't require these.
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=settings.azure_openai_api_version,
    )
    return Reasoner(client, settings.azure_openai_deployment)

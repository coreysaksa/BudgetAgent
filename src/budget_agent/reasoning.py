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
    "rate as a percentage, or null if unknown). It also has spending by category and "
    "income vs. outflow. Answer questions clearly and specifically, citing real numbers "
    "from the snapshot when relevant — reference actual balances and, when comparing debts "
    "or savings, their APRs (e.g. steer extra payments toward the highest-APR balance). "
    "Highlight concrete, prioritized areas where they could improve their budget. Keep "
    "replies concise. If the snapshot is empty or has no accounts, say you don't see any "
    "connected accounts yet and suggest connecting a bank or uploading a statement. You can "
    "explain and advise, but you never move money — transfers require the user's explicit "
    "approval."
)

_PLAN_SYSTEM_PROMPT = (
    "You are BudgetAI, a friendly, careful personal-finance assistant. The user manages "
    "their savings goals by talking to you — this is the ONLY way their goals change.\n\n"
    "You are given (as JSON) the user's current goals, and possibly a snapshot of their "
    "spending analysis. The snapshot includes an `accounts` list (each with `name`, `type`, "
    "`balance`, and `apr` — the annual interest rate as a percentage, or null), plus "
    "spending by category and income vs. outflow. Use real balances and APRs when shaping a "
    "plan: fund goals from available savings/checking balances, account for how much "
    "high-APR debt is costing, and prefer paying down the highest-APR balances. Behave as "
    "follows:\n"
    "- When the user tells you about a goal to add, change, or remove, or asks you to plan "
    "for a goal: write a short, concrete plan in `reply` — reference real numbers from the "
    "snapshot when possible and suggest a realistic monthly contribution and rough "
    "timeline. Then set `goals_updated` to true and return in `goals` the COMPLETE updated "
    "list of goals: keep every existing goal the user did NOT ask to change, apply their "
    "additions and edits, and omit any goal they asked to remove.\n"
    "- When the user is just asking a question and NOT changing goals, set `goals_updated` "
    "to false and return the current goals unchanged in `goals`.\n"
    "- Never invent goals the user didn't ask for. Always pick sensible values: if the user "
    "gives a target amount, estimate a monthly contribution; if they give a monthly amount, "
    "you may leave target_amount at your best estimate or 0.\n\n"
    "Each goal is an object: {\"name\": string, \"target_amount\": number, "
    "\"monthly_contribution\": number or null}. "
    "Respond with ONLY a compact JSON object of the form: "
    "{\"reply\": string, \"goals_updated\": boolean, \"goals\": [goal, ...]}. "
    "Do not wrap it in markdown or add any text outside the JSON."
)


class ChatClient(Protocol):
    """Minimal structural type for an OpenAI-style chat client."""

    @property
    def chat(self) -> Any: ...


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

        goals: list[dict[str, Any]] = []
        for g in goals_raw:
            if not isinstance(g, dict):
                continue
            name = str(g.get("name") or "").strip()
            if not name:
                continue
            try:
                target = float(g.get("target_amount") or 0)
            except (TypeError, ValueError):
                target = 0.0
            monthly = g.get("monthly_contribution")
            try:
                monthly = float(monthly) if monthly is not None else None
            except (TypeError, ValueError):
                monthly = None
            goals.append(
                {
                    "name": name,
                    "target_amount": target,
                    "monthly_contribution": monthly,
                }
            )
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

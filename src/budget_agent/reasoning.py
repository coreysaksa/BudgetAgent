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
from datetime import date
from typing import Any, Protocol

from .approval import MoneyAction
from .config import Settings
from .models import BudgetPlan
from .prompts import (
    ADJUDICATE_SYSTEM_PROMPT as _ADJUDICATE_SYSTEM_PROMPT,
    CASH_FLOW_INPUT_SYSTEM_PROMPT as _CASH_FLOW_INPUT_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT as _CHAT_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT as _PLAN_SYSTEM_PROMPT,
    SYSTEM_PROMPT as _SYSTEM_PROMPT,
)


class ChatClient(Protocol):
    """Minimal structural type for an OpenAI-style chat client."""

    @property
    def chat(self) -> Any: ...


_MISSING = object()
_GOAL_KINDS = {"savings", "debt_payoff", "milestone", "purchase"}
_PAYMENT_TIMINGS = {"upfront", "at_checkout"}
_HORIZONS = {"short", "mid", "long"}
_DEADLINE_TYPES = {"hard", "soft", "none"}
_GOAL_STATUSES = {"active", "paused", "completed"}


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
    horizon = pick("horizon", lambda v: str(v or "mid").strip().lower(), "mid")
    if horizon not in _HORIZONS:
        horizon = "mid"
    deadline_type = pick(
        "deadline_type", lambda v: str(v or "soft").strip().lower(), "soft"
    )
    if deadline_type not in _DEADLINE_TYPES:
        deadline_type = "soft"
    status = pick("status", lambda v: str(v or "active").strip().lower(), "active")
    if status not in _GOAL_STATUSES:
        status = "active"

    return {
        "id": _to_str(g.get("id")) or _to_str(prior.get("id")),
        "name": str(g.get("name") or "").strip(),
        "kind": kind,
        "target_amount": pick("target_amount", _to_float, None),
        "target_date": pick("target_date", _to_str, None),
        "monthly_contribution": pick("monthly_contribution", _to_float, None),
        "priority": pick(
            "priority",
            lambda v: min(5, max(1, int(_to_float(v) or 3))),
            3,
        ),
        "horizon": horizon,
        "deadline_type": deadline_type,
        "minimum_monthly": pick("minimum_monthly", _to_float, None),
        "status": status,
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

    def adjudicate_merchants(
        self, merchant: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Judge whether each candidate merchant name is the same business.

        Used to auto-resolve borderline fuzzy matches when the user recategorizes a
        merchant: for each candidate the model decides if it's the *same* business
        (e.g. a misspelling or a payment-processor alias). Returns
        ``{"decisions": [{"merchant": str, "same": bool}, ...]}``.
        """
        payload = {"merchant": merchant, "candidates": candidates}
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": _ADJUDICATE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        return {"decisions": self._parse_decisions(raw)}

    def extract_cash_flow_inputs(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Extract explicit windfalls, pay dates, and necessity answers for the planner."""
        conversation = [
            turn
            for turn in (history or [])
            if turn.get("role") in {"user", "assistant"} and turn.get("content")
        ]
        conversation.append({"role": "user", "content": message})
        payload = {
            "current_date": date.today().isoformat(),
            "conversation": conversation,
        }
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": _CASH_FLOW_INPUT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_format={"type": "json_object"},
        )
        return self._parse_cash_flow_inputs(resp.choices[0].message.content or "")

    @staticmethod
    def _parse_decisions(raw: str) -> list[dict[str, Any]]:
        """Validate the model's same-merchant decisions, ignoring malformed rows."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        rows = data.get("decisions") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = str(item.get("merchant") or "").strip()
            if name:
                out.append({"merchant": name, "same": bool(item.get("same"))})
        return out

    @staticmethod
    def _parse_cash_flow_inputs(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = {}
        if not isinstance(data, dict):
            data = {}

        windfalls: list[dict[str, Any]] = []
        for item in data.get("windfalls") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            amount = _to_float(item.get("amount"))
            when = _to_str(item.get("date"))
            status = str(item.get("status") or "estimated").strip().lower()
            try:
                date.fromisoformat(when or "")
            except ValueError:
                when = None
            if name and amount and amount > 0 and when:
                windfalls.append(
                    {
                        "name": name,
                        "amount": amount,
                        "date": when,
                        "status": status if status in {"confirmed", "estimated"} else "estimated",
                    }
                )

        paychecks: list[dict[str, Any]] = []
        for item in data.get("paychecks") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "Paycheck").strip() or "Paycheck"
            amount = _to_float(item.get("amount"))
            try:
                day = int(item.get("day"))
            except (TypeError, ValueError):
                day = 0
            if amount and amount > 0 and 1 <= day <= 31:
                paychecks.append({"name": name, "amount": amount, "day": day})

        necessity: list[dict[str, str]] = []
        for item in data.get("necessity_overrides") or []:
            if not isinstance(item, dict):
                continue
            merchant = str(item.get("merchant") or "").strip()
            value = str(item.get("necessity") or "").strip().lower()
            if merchant and value in {"mandatory", "discretionary"}:
                necessity.append({"merchant": merchant, "necessity": value})

        clarifications: list[str] = []
        for item in data.get("clarifications") or []:
            question = (
                str(item.get("question") or "").strip()
                if isinstance(item, dict)
                else str(item or "").strip()
            )
            if question:
                clarifications.append(question)
        return {
            "windfalls": windfalls,
            "paychecks": paychecks,
            "necessity_overrides": necessity,
            "clarifications": clarifications,
        }

    @staticmethod
    def _parse_plan_response(
        raw: str, current_goals: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Parse the model's JSON reply, degrading gracefully on malformed output."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Fall back to treating the whole text as a plain reply, changing nothing.
            return {
                "reply": raw,
                "goals_updated": False,
                "goals": current_goals,
            }

        reply = str(data.get("reply") or "")
        goals_updated = bool(data.get("goals_updated"))
        goals_raw = data.get("goals")
        if not goals_updated or not isinstance(goals_raw, list):
            return {
                "reply": reply,
                "goals_updated": False,
                "goals": current_goals,
            }

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
        return {
            "reply": reply,
            "goals_updated": True,
            "goals": goals,
        }


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
        # The deployment has a modest TPM budget, so brief bursts can hit 429s.
        # Let the SDK retry with exponential backoff (honoring Retry-After)
        # before we surface a rate-limit error to the caller.
        max_retries=6,
    )
    return Reasoner(client, settings.azure_openai_deployment)

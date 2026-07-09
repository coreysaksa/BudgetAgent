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
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


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

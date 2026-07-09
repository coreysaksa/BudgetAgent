"""Runtime configuration for the orchestrator."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


@dataclass
class Settings:
    require_approval: bool = True
    auto_topup_cap: float = 0.0
    key_vault_uri: str = ""
    aggregator_url: str = "http://localhost:8001"
    analyzer_url: str = "http://localhost:8002"
    planner_url: str = "http://localhost:8003"
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o-mini"
    azure_openai_api_version: str = "2024-10-21"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            require_approval=_bool("REQUIRE_APPROVAL", True),
            auto_topup_cap=float(os.getenv("AUTO_TOPUP_CAP", "0")),
            key_vault_uri=os.getenv("AZURE_KEY_VAULT_URI", ""),
            aggregator_url=os.getenv("AGGREGATOR_URL", "http://localhost:8001"),
            analyzer_url=os.getenv("ANALYZER_URL", "http://localhost:8002"),
            planner_url=os.getenv("PLANNER_URL", "http://localhost:8003"),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

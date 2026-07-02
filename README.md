# BudgetAgent

Agentic AI orchestrator for **BudgetAI** — it analyzes your spending, builds a budget from
your goals, and drives execution against that budget. It is the "brain" that coordinates the
separate tool repos.

## System architecture

```
                         ┌─────────────────────────┐
                         │      BudgetAgent         │   (this repo, orchestrator)
                         │  analyze → plan →        │
                         │  propose → APPROVE →     │
                         │  execute → track         │
                         └───────────┬─────────────┘
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
  budget-tool-aggregator   budget-tool-analyzer    budget-tool-planner
  (Plaid, read-only)       (categorize spending)   (goals → budget plan)
             │                       │                       │
             └───────────────────────┴───────────────────────┘
                                     │
                             budget-infra (Azure)
                    Key Vault (creds) · Storage (plan+tracking)
                                     │
                                budget-web (React dashboard)
```

## Repos

| Repo | Role |
|------|------|
| **BudgetAgent** (this) | Orchestrator / agent brain |
| `budget-tool-aggregator` | Pull transactions & balances (bank/credit/mortgage/utility) via Plaid, provider-swappable |
| `budget-tool-analyzer` | Categorize & analyze spending, incl. the petty-cash checking account |
| `budget-tool-planner` | Generate a budget plan from your goals |
| `budget-web` | Visualize progress toward goals |
| `budget-infra` | Azure IaC (Bicep): Key Vault, Storage, web hosting, managed identity |

## The execution lifecycle

1. **Analyze** — aggregate transactions (aggregator tool) → categorize & summarize (analyzer tool).
2. **Plan** — generate a budget from your goals (planner tool).
3. **Propose** — present the plan and any recommended money movements.
4. **Approve** ⛔ — a **human approval gate**. See Safety model.
5. **Execute** — apply approved actions; automatically **track** progress.

## ⚠️ Safety model (read this)

- Account aggregation (Plaid/MX/Finicity) is **read-only**. It can pull balances and
  transactions but **cannot move money**.
- **Autonomous money movement is heavily regulated and risky.** This system is designed so
  that anything that moves money is **approval-gated** by default (`REQUIRE_APPROVAL=true`).
  The agent may *recommend* transfers; a human confirms them until you explicitly opt into a
  narrower, well-tested auto-execute policy (e.g. capped auto-top-ups to the petty-cash
  checking account).
- Fully automated with **no gate**: only *analysis, planning, and progress tracking*.
- Credentials (Plaid keys, bank tokens) live in **Azure Key Vault** (see `budget-infra`),
  never in source control.

## The "petty cash" account

One checking account is designated **petty cash** — personal discretionary spending
(groceries, dining out, clothes). The planner allocates a periodic amount to it; the analyzer
tracks burn-down against that allocation.

## Layout

```
src/budget_agent/
  orchestrator.py   # analyze→plan→propose→approve→execute state machine
  approval.py       # human-approval gate for money-moving actions
  tools.py          # typed clients for the aggregator/analyzer/planner tools
  config.py         # settings (Key Vault refs, approval policy)
  models.py         # shared domain models
tests/
```

## Status

Scaffold. See the project board (Project #1) for the task plan.

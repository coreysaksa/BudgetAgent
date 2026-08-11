from budget_agent.prompts import (
    CASH_FLOW_INPUT_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
)


def test_prompts_treat_keep_the_change_as_transfer():
    for prompt in (CHAT_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT):
        assert "Keep the Change" in prompt
        assert "never income" in prompt.lower()
        assert "transfer" in prompt.lower()


def test_prompts_use_authoritative_cash_flow_plan():
    for prompt in (CHAT_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT):
        assert "cash_flow_plan" in prompt
        assert "deterministic" in prompt.lower()
        assert "estimated windfall" in prompt.lower()


def test_cash_flow_input_prompt_requires_explicit_facts():
    prompt = CASH_FLOW_INPUT_SYSTEM_PROMPT.lower()
    assert "do not estimate or invent" in prompt
    assert "sca" in prompt
    assert "paychecks" in prompt
    assert "necessity_overrides" in prompt

from budget_agent.prompts import CHAT_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT


def test_prompts_treat_keep_the_change_as_transfer():
    for prompt in (CHAT_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT):
        assert "Keep the Change" in prompt
        assert "never income" in prompt.lower()
        assert "transfer" in prompt.lower()

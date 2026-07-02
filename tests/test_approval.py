from budget_agent.approval import ApprovalPolicy, ApprovalRequired, MoneyAction


def _action(kind="transfer", amount=100.0):
    return MoneyAction(kind=kind, amount=amount, source_account_id="a",
                       dest_account_id="b", reason="test")


def test_transfer_requires_approval_by_default():
    policy = ApprovalPolicy(require_approval=True)
    try:
        policy.guard(_action())
        assert False, "expected ApprovalRequired"
    except ApprovalRequired:
        pass


def test_human_approval_allows_action():
    ApprovalPolicy(require_approval=True).guard(_action(), human_approved=True)


def test_capped_petty_cash_topup_auto_allowed():
    policy = ApprovalPolicy(require_approval=True, auto_topup_cap=200.0)
    policy.guard(_action(kind="petty_cash_topup", amount=150.0))


def test_over_cap_topup_still_requires_approval():
    policy = ApprovalPolicy(require_approval=True, auto_topup_cap=100.0)
    try:
        policy.guard(_action(kind="petty_cash_topup", amount=150.0))
        assert False, "expected ApprovalRequired"
    except ApprovalRequired:
        pass

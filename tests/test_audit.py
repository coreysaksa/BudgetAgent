from budget_agent.approval import ApprovalPolicy, ApprovalRequired, MoneyAction
from budget_agent.audit import AuditLog


def _action(kind="transfer", amount=100.0):
    return MoneyAction(kind=kind, amount=amount, source_account_id="a",
                       dest_account_id="b", reason="test")


def test_records_human_approved():
    log = AuditLog()
    ApprovalPolicy(require_approval=True).guard(_action(), human_approved=True, audit=log)
    assert len(log.entries) == 1
    assert log.entries[0].decision == "human_approved"
    assert log.entries[0].amount == 100.0


def test_records_auto_approved():
    log = AuditLog()
    policy = ApprovalPolicy(require_approval=True, auto_topup_cap=200.0)
    policy.guard(_action(kind="petty_cash_topup", amount=150.0), audit=log)
    assert log.entries[0].decision == "auto_approved"


def test_records_denied_and_raises():
    log = AuditLog()
    try:
        ApprovalPolicy(require_approval=True).guard(_action(), audit=log)
        assert False, "expected ApprovalRequired"
    except ApprovalRequired:
        pass
    assert log.entries[0].decision == "denied"


def test_log_is_append_only_and_ordered():
    log = AuditLog()
    policy = ApprovalPolicy(require_approval=True, auto_topup_cap=200.0)
    policy.guard(_action(kind="petty_cash_topup", amount=50.0), audit=log)
    policy.guard(_action(), human_approved=True, audit=log)
    decisions = [e.decision for e in log.entries]
    assert decisions == ["auto_approved", "human_approved"]


def test_record_rejects_unknown_decision():
    log = AuditLog()
    try:
        log.record(_action(), "maybe")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_guard_without_audit_still_works():
    ApprovalPolicy(require_approval=True).guard(_action(), human_approved=True)

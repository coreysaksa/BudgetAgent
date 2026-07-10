"""Notifications for proposed money actions awaiting human approval.

When the orchestrator proposes actions (recommend flow), a notification is dispatched
so the appropriate human can review and approve them before any money moves. The
notifier is a no-op when no webhook URL is configured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .approval import MoneyAction


@dataclass
class NotificationPayload:
    """Payload sent to the notification webhook."""

    event: str
    period: str
    actions: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "period": self.period,
            "actions": self.actions,
        }


class Notifier:
    """Sends webhook notifications about proposed actions awaiting approval.

    If ``webhook_url`` is empty the notifier is a no-op, so callers need not
    check whether notifications are configured before calling ``notify_proposed``.
    """

    def __init__(
        self,
        webhook_url: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self._webhook_url)

    def notify_proposed(self, actions: list["MoneyAction"], period: str) -> None:
        """Notify that ``actions`` have been proposed and await approval.

        Sends a single HTTP POST to the configured webhook with the action list
        so a human reviewer can inspect and approve them. If no webhook URL is
        configured, this is a no-op.

        Raises ``httpx.HTTPStatusError`` if the webhook returns a non-2xx status.
        """
        if not self._webhook_url:
            return
        payload = NotificationPayload(
            event="actions_proposed",
            period=period,
            actions=[
                {
                    "kind": a.kind,
                    "amount": a.amount,
                    "source_account_id": a.source_account_id,
                    "dest_account_id": a.dest_account_id,
                    "reason": a.reason,
                }
                for a in actions
            ],
        )
        client = httpx.Client(transport=self._transport)
        try:
            resp = client.post(self._webhook_url, json=payload.as_dict())
            resp.raise_for_status()
        finally:
            client.close()

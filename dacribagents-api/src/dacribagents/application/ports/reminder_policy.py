"""Port for reminder delivery policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from dacribagents.domain.reminders.enums import DeliveryChannel, UrgencyLevel


@dataclass(frozen=True)
class ChannelSelection:
    """Result of policy evaluation for a single member."""

    member_id: UUID
    primary_channel: DeliveryChannel
    primary_address: str
    fallback_channel: DeliveryChannel | None = None
    fallback_address: str | None = None
    suppressed: bool = False
    suppression_reason: str | None = None


class ReminderPolicy(Protocol):
    """Evaluates delivery policy for a reminder targeting one or more members.

    Considers: PreferenceRules, quiet hours, urgency, member contact info.
    """

    def select_channels(
        self,
        household_id: UUID,
        member_ids: list[UUID],
        urgency: UrgencyLevel,
    ) -> list[ChannelSelection]:
        """Determine delivery channel(s) for each target member.

        Returns one ChannelSelection per member. Members in quiet hours with
        non-critical urgency will have ``suppressed=True``.
        """
        ...

    def check_rate_limit(
        self,
        member_id: UUID,
        daily_count: int,
    ) -> bool:
        """Return True if the member can receive another reminder today."""
        ...

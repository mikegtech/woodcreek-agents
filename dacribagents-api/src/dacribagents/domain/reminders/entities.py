"""Frozen dataclass entities for the household reminder domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Mapping
from uuid import UUID

from dacribagents.domain.reminders.enums import (
    AckMethod,
    ApprovalAction,
    CalendarAccessLevel,
    CalendarProviderType,
    DeliveryChannel,
    DeliveryStatus,
    DevicePlatform,
    MemberRole,
    NotificationIntent,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)


@dataclass(frozen=True)
class Household:
    """Top-level household unit. One per deployment."""

    id: UUID
    name: str
    created_at: datetime
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HouseholdMember:
    """A person in the household with contact info and delivery preferences."""

    id: UUID
    household_id: UUID
    name: str
    role: MemberRole
    timezone: str
    created_at: datetime
    email: str | None = None
    phone: str | None = None
    slack_id: str | None = None
    quiet_hours_start: time | None = None
    quiet_hours_end: time | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Reminder:
    """The core reminder entity. Tracks subject, urgency, source, and lifecycle state."""

    id: UUID
    household_id: UUID
    subject: str
    body: str
    urgency: UrgencyLevel
    source: ReminderSource
    state: ReminderState
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    intent: NotificationIntent = NotificationIntent.REMINDER
    source_agent: str | None = None
    source_event_id: str | None = None
    dedupe_key: str | None = None
    requires_approval: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReminderTarget:
    """Junction linking a reminder to its intended recipients."""

    id: UUID
    reminder_id: UUID
    target_type: TargetType
    member_id: UUID | None = None
    role: MemberRole | None = None


@dataclass(frozen=True)
class ReminderSchedule:
    """When a reminder should fire. Supports one-shot, relative, and recurring."""

    id: UUID
    reminder_id: UUID
    schedule_type: ScheduleType
    timezone: str
    fire_at: datetime | None = None
    relative_to: str | None = None
    relative_offset_minutes: int | None = None
    cron_expression: str | None = None
    next_fire_at: datetime | None = None


@dataclass(frozen=True)
class ReminderExecution:
    """Runtime record of a scheduled reminder firing."""

    id: UUID
    reminder_id: UUID
    schedule_id: UUID
    fired_at: datetime
    created_at: datetime


@dataclass(frozen=True)
class ReminderDelivery:
    """Record of a delivery attempt to a specific member via a specific channel."""

    id: UUID
    execution_id: UUID
    member_id: UUID
    channel: DeliveryChannel
    status: DeliveryStatus
    created_at: datetime
    escalation_step: int = 0
    provider_message_id: str | None = None
    sent_at: datetime | None = None
    delivered_at: datetime | None = None
    failed_at: datetime | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class ReminderAcknowledgement:
    """A member's response to a delivered reminder (ack or snooze)."""

    id: UUID
    delivery_id: UUID
    member_id: UUID
    method: AckMethod
    acknowledged_at: datetime
    note: str | None = None
    snoozed_until: datetime | None = None


@dataclass(frozen=True)
class ApprovalRecord:
    """Audit record for approval lifecycle actions (submit, approve, reject)."""

    id: UUID
    reminder_id: UUID
    action: ApprovalAction
    actor_id: UUID
    created_at: datetime
    reason: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PreferenceRule:
    """Per-member or household-wide delivery preference rule."""

    id: UUID
    household_id: UUID
    preferred_channel: DeliveryChannel
    active: bool
    created_at: datetime
    member_id: UUID | None = None
    urgency: UrgencyLevel | None = None
    fallback_channel: DeliveryChannel | None = None
    quiet_hours_override: bool = False


@dataclass(frozen=True)
class ReminderContextSource:
    """An external data source providing reminder-relevant context."""

    id: UUID
    household_id: UUID
    source_type: str
    source_ref: str
    title: str
    created_at: datetime
    event_date: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CalendarIdentity:
    """A calendar account linked to a member or the household."""

    id: UUID
    household_id: UUID
    provider: CalendarProviderType
    provider_account_id: str
    display_name: str
    active: bool
    created_at: datetime
    member_id: UUID | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CalendarProvider:
    """A supported calendar backend and its configuration."""

    id: UUID
    provider_type: CalendarProviderType
    display_name: str
    active: bool
    config: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CalendarAccessPolicy:
    """Per-identity rules for calendar access: read/write, visible calendars, sync."""

    id: UUID
    calendar_identity_id: UUID
    access_level: CalendarAccessLevel
    sync_frequency_minutes: int
    created_at: datetime
    visible_calendars: tuple[str, ...] = ()
    write_back_enabled: bool = False


@dataclass(frozen=True)
class MemberDevice:
    """Registered device for push notifications (anticipated for future APNs)."""

    id: UUID
    member_id: UUID
    platform: DevicePlatform
    device_token: str
    active: bool
    registered_at: datetime
    display_name: str | None = None

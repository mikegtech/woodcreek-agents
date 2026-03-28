"""Pydantic models for reminder API contracts (request/response)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from dacribagents.domain.reminders.enums import (
    AckMethod,
    DeliveryChannel,
    MemberRole,
    NotificationIntent,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)

# ── Request models ──────────────────────────────────────────────────────────


class ReminderTargetInput(BaseModel):
    """Specifies who a reminder targets."""

    target_type: TargetType
    member_id: UUID | None = None
    role: MemberRole | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> ReminderTargetInput:
        if self.target_type == TargetType.INDIVIDUAL and self.member_id is None:
            raise ValueError("member_id is required for INDIVIDUAL targeting")
        if self.target_type == TargetType.ROLE and self.role is None:
            raise ValueError("role is required for ROLE targeting")
        if self.target_type == TargetType.HOUSEHOLD and (self.member_id is not None or self.role is not None):
            raise ValueError("member_id and role must be None for HOUSEHOLD targeting")
        return self


class ReminderScheduleInput(BaseModel):
    """Specifies when a reminder fires."""

    schedule_type: ScheduleType
    timezone: str = "America/Chicago"
    fire_at: datetime | None = None
    relative_to: str | None = None
    relative_offset_minutes: int | None = None
    cron_expression: str | None = None

    @model_validator(mode="after")
    def _validate_schedule(self) -> ReminderScheduleInput:
        if self.schedule_type == ScheduleType.ONE_SHOT and self.fire_at is None:
            raise ValueError("fire_at is required for ONE_SHOT schedule")
        if self.schedule_type == ScheduleType.RELATIVE:
            if self.relative_to is None or self.relative_offset_minutes is None:
                raise ValueError("relative_to and relative_offset_minutes are required for RELATIVE schedule")
        if self.schedule_type == ScheduleType.RECURRING and self.cron_expression is None:
            raise ValueError("cron_expression is required for RECURRING schedule")
        return self


class CreateReminderRequest(BaseModel):
    """Create a new reminder draft."""

    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=5000)
    urgency: UrgencyLevel = UrgencyLevel.NORMAL
    intent: NotificationIntent = NotificationIntent.REMINDER
    source: ReminderSource = ReminderSource.USER
    source_agent: str | None = None
    source_event_id: str | None = None
    dedupe_key: str | None = None
    targets: list[ReminderTargetInput] = Field(min_length=1)
    schedule: ReminderScheduleInput | None = None
    requires_approval: bool = False

    @model_validator(mode="after")
    def _validate_source(self) -> CreateReminderRequest:
        if self.source == ReminderSource.AGENT and not self.source_agent:
            raise ValueError("source_agent is required when source is AGENT")
        return self


class UpdateReminderRequest(BaseModel):
    """Update fields on a draft reminder. Only mutable in DRAFT state."""

    subject: str | None = Field(default=None, min_length=1, max_length=500)
    body: str | None = Field(default=None, max_length=5000)
    urgency: UrgencyLevel | None = None
    targets: list[ReminderTargetInput] | None = Field(default=None, min_length=1)
    schedule: ReminderScheduleInput | None = None


class ScheduleReminderRequest(BaseModel):
    """Attach a schedule to a draft and transition to SCHEDULED."""

    schedule: ReminderScheduleInput


class AcknowledgeReminderRequest(BaseModel):
    """Acknowledge a delivered reminder."""

    member_id: UUID
    method: AckMethod
    note: str | None = None


class SnoozeReminderRequest(BaseModel):
    """Snooze a delivered reminder for a duration."""

    member_id: UUID
    method: AckMethod
    snooze_until: datetime
    note: str | None = None


class EventIntakeRequest(BaseModel):
    """Intake a reminder from an upstream event source.

    This is the seam between event-producing services (solar telemetry,
    maintenance agents, HOA compliance, warranty tracking, IoT sensors)
    and the reminder platform.  All event-originated reminders require
    approval — no autonomous delivery in current phases.
    """

    source: ReminderSource
    source_event_id: str = Field(min_length=1)
    dedupe_key: str | None = None
    household_id: UUID
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=5000)
    urgency: UrgencyLevel = UrgencyLevel.NORMAL
    intent: NotificationIntent = NotificationIntent.ALERT
    source_agent: str | None = None
    targets: list[ReminderTargetInput] = Field(min_length=1)
    schedule: ReminderScheduleInput | None = None


# ── Response models ─────────────────────────────────────────────────────────


class ReminderTargetResponse(BaseModel):
    """Serialized reminder target."""

    id: UUID
    reminder_id: UUID
    target_type: TargetType
    member_id: UUID | None = None
    role: MemberRole | None = None


class ReminderScheduleResponse(BaseModel):
    """Serialized reminder schedule."""

    id: UUID
    reminder_id: UUID
    schedule_type: ScheduleType
    timezone: str
    fire_at: datetime | None = None
    relative_to: str | None = None
    relative_offset_minutes: int | None = None
    cron_expression: str | None = None
    next_fire_at: datetime | None = None


class ReminderResponse(BaseModel):
    """Serialized reminder with targets and schedule."""

    id: UUID
    household_id: UUID
    subject: str
    body: str
    urgency: UrgencyLevel
    intent: NotificationIntent
    source: ReminderSource
    source_agent: str | None
    source_event_id: str | None
    dedupe_key: str | None
    state: ReminderState
    requires_approval: bool
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    targets: list[ReminderTargetResponse] = Field(default_factory=list)
    schedule: ReminderScheduleResponse | None = None


class ReminderListResponse(BaseModel):
    """Paginated list of reminders."""

    items: list[ReminderResponse]
    total: int
    offset: int
    limit: int


class DeliveryResponse(BaseModel):
    """Serialized delivery attempt."""

    id: UUID
    execution_id: UUID
    member_id: UUID
    channel: DeliveryChannel
    status: str
    escalation_step: int
    provider_message_id: str | None = None
    sent_at: datetime | None = None
    delivered_at: datetime | None = None
    failure_reason: str | None = None

"""Enums for the household reminder orchestration domain."""

from __future__ import annotations

from enum import Enum


class ReminderState(str, Enum):
    """Lifecycle states for a reminder."""

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SCHEDULED = "scheduled"
    PENDING_DELIVERY = "pending_delivery"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    SNOOZED = "snoozed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ApprovalAction(str, Enum):
    """Actions recorded in the approval audit trail."""

    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"


class UrgencyLevel(str, Enum):
    """Urgency classification for reminder delivery and escalation."""

    CRITICAL = "critical"
    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"


class DeliveryChannel(str, Enum):
    """Supported delivery channels for reminders."""

    PUSH = "push"
    SMS = "sms"
    EMAIL = "email"
    SLACK = "slack"
    IN_APP = "in_app"


class DeliveryStatus(str, Enum):
    """Status of a single delivery attempt."""

    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    BOUNCED = "bounced"
    FAILED = "failed"


class AckMethod(str, Enum):
    """How a household member acknowledged a reminder."""

    SLACK_REACTION = "slack_reaction"
    SLACK_BUTTON = "slack_button"
    SMS_REPLY = "sms_reply"
    PUSH_ACTION = "push_action"
    EMAIL_LINK = "email_link"
    DASHBOARD = "dashboard"


class MemberRole(str, Enum):
    """Roles within a household."""

    ADMIN = "admin"
    MEMBER = "member"
    CHILD = "child"


class TargetType(str, Enum):
    """How a reminder targets recipients."""

    INDIVIDUAL = "individual"
    HOUSEHOLD = "household"
    ROLE = "role"


class ScheduleType(str, Enum):
    """How a reminder schedule fires."""

    ONE_SHOT = "one_shot"
    RELATIVE = "relative"
    RECURRING = "recurring"


class CalendarProviderType(str, Enum):
    """Supported calendar backend providers."""

    WORKMAIL_EWS = "workmail_ews"
    ICAL_FEED = "ical_feed"
    GOOGLE = "google"
    CALDAV = "caldav"
    MANUAL = "manual"


class CalendarAccessLevel(str, Enum):
    """What the platform can do with a calendar identity."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class DevicePlatform(str, Enum):
    """Mobile/device platforms for push delivery."""

    IOS = "ios"


class NotificationIntent(str, Enum):
    """What kind of notification this reminder represents."""

    REMINDER = "reminder"  # "Don't forget to do X by Y"
    ALERT = "alert"  # "Something happened that needs attention now"
    DIGEST = "digest"  # "Here's a summary of recent activity"


class ReminderSource(str, Enum):
    """Where a reminder originated.

    Domain-specific values (maintenance, hoa, warranty, telemetry) identify
    the upstream producer directly.  ``agent`` is the fallback for agents
    without a dedicated source category — set ``source_agent`` to distinguish.
    """

    USER = "user"
    AGENT = "agent"
    CALENDAR = "calendar"
    EMAIL = "email"
    MAINTENANCE = "maintenance"
    HOA = "hoa"
    WARRANTY = "warranty"
    TELEMETRY = "telemetry"
    HOUSEHOLD_ROUTINE = "household_routine"
    EVENT_BUS = "event_bus"

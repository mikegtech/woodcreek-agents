"""Shared fixtures for reminder domain tests."""

from __future__ import annotations

from datetime import datetime, time
from uuid import uuid4

import pytest

from dacribagents.domain.reminders.entities import (
    Household,
    HouseholdMember,
    Reminder,
    ReminderSchedule,
    ReminderTarget,
)
from dacribagents.domain.reminders.enums import (
    MemberRole,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)


@pytest.fixture()
def household() -> Household:
    return Household(
        id=uuid4(),
        name="Woodcreek Household",
        created_at=datetime(2026, 1, 1),
    )


@pytest.fixture()
def admin_member(household: Household) -> HouseholdMember:
    return HouseholdMember(
        id=uuid4(),
        household_id=household.id,
        name="Mike",
        role=MemberRole.ADMIN,
        timezone="America/Chicago",
        created_at=datetime(2026, 1, 1),
        email="mike@woodcreek.me",
        phone="+15125551234",
        slack_id="U1234MIKE",
        quiet_hours_start=time(22, 0),
        quiet_hours_end=time(7, 0),
    )


@pytest.fixture()
def child_member(household: Household) -> HouseholdMember:
    return HouseholdMember(
        id=uuid4(),
        household_id=household.id,
        name="Kid",
        role=MemberRole.CHILD,
        timezone="America/Chicago",
        created_at=datetime(2026, 1, 1),
    )


@pytest.fixture()
def draft_reminder(household: Household, admin_member: HouseholdMember) -> Reminder:
    return Reminder(
        id=uuid4(),
        household_id=household.id,
        subject="Take out trash",
        body="Bins to curb by 7am Wednesday",
        urgency=UrgencyLevel.NORMAL,
        source=ReminderSource.USER,
        state=ReminderState.DRAFT,
        created_by=admin_member.id,
        created_at=datetime(2026, 3, 28, 10, 0),
        updated_at=datetime(2026, 3, 28, 10, 0),
    )


@pytest.fixture()
def household_target(draft_reminder: Reminder) -> ReminderTarget:
    return ReminderTarget(
        id=uuid4(),
        reminder_id=draft_reminder.id,
        target_type=TargetType.HOUSEHOLD,
    )


@pytest.fixture()
def individual_target(draft_reminder: Reminder, admin_member: HouseholdMember) -> ReminderTarget:
    return ReminderTarget(
        id=uuid4(),
        reminder_id=draft_reminder.id,
        target_type=TargetType.INDIVIDUAL,
        member_id=admin_member.id,
    )


@pytest.fixture()
def one_shot_schedule(draft_reminder: Reminder) -> ReminderSchedule:
    return ReminderSchedule(
        id=uuid4(),
        reminder_id=draft_reminder.id,
        schedule_type=ScheduleType.ONE_SHOT,
        timezone="America/Chicago",
        fire_at=datetime(2026, 4, 2, 6, 30),
        next_fire_at=datetime(2026, 4, 2, 6, 30),
    )

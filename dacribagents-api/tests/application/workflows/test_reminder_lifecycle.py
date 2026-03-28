"""Tests for LangGraph reminder lifecycle workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.application.services.governance import AutonomyTier, GovernanceState, get_governance_state
from dacribagents.application.use_cases.reminder_workflows import (
    approve_reminder,
    create_reminder,
    schedule_reminder,
    submit_for_approval,
)
from dacribagents.application.workflows.reminder_lifecycle import (
    build_reminder_graph,
    initial_state,
)
from dacribagents.application.workflows.runner import run_single_workflow, run_workflows
from dacribagents.domain.reminders.entities import Household, HouseholdMember
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    DeliveryStatus,
    MemberRole,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
)
from dacribagents.domain.reminders.models import (
    ApproveReminderRequest,
    CreateReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    ScheduleReminderRequest,
    SubmitForApprovalRequest,
)
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore


class StubEmailAdapter:
    channel = DeliveryChannel.EMAIL

    def send(self, **kw) -> DeliveryResult:
        return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id="stub")


@pytest.fixture(autouse=True)
def _reset_gov():
    import dacribagents.application.services.governance as mod

    mod._state = GovernanceState()
    yield
    mod._state = None


@pytest.fixture()
def household_id():
    return uuid4()


@pytest.fixture()
def member_id():
    return uuid4()


@pytest.fixture()
def store(household_id, member_id):
    s = InMemoryReminderStore()
    s.households[household_id] = Household(id=household_id, name="Woodcreek", created_at=datetime(2026, 1, 1))
    s.members[member_id] = HouseholdMember(
        id=member_id, household_id=household_id, name="Mike",
        role=MemberRole.ADMIN, timezone="America/Chicago",
        created_at=datetime(2026, 1, 1), email="mike@test.com",
    )
    return s


# ── Graph build and basic execution ────────────────────────────────────────


def test_graph_builds():
    store = InMemoryReminderStore()
    graph = build_reminder_graph(store)
    assert graph is not None


def test_graph_nonexistent_reminder(store, household_id):
    graph = build_reminder_graph(store)
    state = initial_state(uuid4(), household_id)
    result = graph.invoke(state, {"configurable": {"thread_id": "test-1"}})
    assert result["outcome"] == "not_found"


def test_graph_draft_no_approval(store, household_id, member_id):
    """User-created draft without approval → check_schedule (waiting)."""
    get_governance_state().set_tier(household_id, AutonomyTier.TIER_1)

    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Test", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    result = run_single_workflow(store, {}, reminder.id, household_id)
    assert result is not None
    assert result["outcome"] == "waiting"  # DRAFT, not yet scheduled


def test_graph_approval_wait_and_resume(store, household_id, member_id):
    """HOA reminder → waiting (pending approval) → approve → still waiting (needs schedule)."""
    get_governance_state().set_tier(household_id, AutonomyTier.TIER_1)

    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="HOA dues", source=ReminderSource.HOA, source_event_id="hoa:test",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    submit_for_approval(store, reminder.id, SubmitForApprovalRequest(actor_id=member_id))

    # First run: pending approval → waiting
    result1 = run_single_workflow(store, {}, reminder.id, household_id)
    assert result1["outcome"] == "waiting"

    # Approve via the store (simulating Slack approval)
    approve_reminder(store, reminder.id, ApproveReminderRequest(actor_id=member_id))

    # Resume: approved but not scheduled → still waiting
    result2 = run_single_workflow(store, {}, reminder.id, household_id)
    assert result2["outcome"] == "waiting"


def test_graph_full_delivery(store, household_id, member_id):
    """Scheduled → tick → PENDING_DELIVERY → graph delivers → DELIVERED."""
    get_governance_state().set_tier(household_id, AutonomyTier.TIER_1)

    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Delivery test",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            schedule=ReminderScheduleInput(
                schedule_type=ScheduleType.ONE_SHOT,
                fire_at=datetime(2026, 4, 1, tzinfo=UTC),
            ),
        ),
    )
    # Advance to PENDING_DELIVERY
    store.update_reminder_state(reminder.id, ReminderState.PENDING_DELIVERY)

    adapter = StubEmailAdapter()
    result = run_single_workflow(
        store, {DeliveryChannel.EMAIL: adapter}, reminder.id, household_id,
    )
    assert result is not None
    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.DELIVERED
    assert result["outcome"] == "delivered"


def test_graph_rejected_terminates(store, household_id, member_id):
    """Rejected reminder → complete(rejected)."""
    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Reject test", source=ReminderSource.HOA, source_event_id="hoa:r",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    store.update_reminder_state(reminder.id, ReminderState.PENDING_APPROVAL)
    store.update_reminder_state(reminder.id, ReminderState.REJECTED)

    result = run_single_workflow(store, {}, reminder.id, household_id)
    assert result["outcome"] == "rejected"


def test_graph_cancelled_terminates(store, household_id, member_id):
    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Cancel test", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    store.update_reminder_state(reminder.id, ReminderState.CANCELLED)

    result = run_single_workflow(store, {}, reminder.id, household_id)
    assert result["outcome"] == "cancelled"


# ── Runner integration with control loop ────────────────────────────────────


def test_run_workflows_processes_active(store, household_id, member_id):
    get_governance_state().set_tier(household_id, AutonomyTier.TIER_1)

    create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="A", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="B", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )

    count = run_workflows(store, {}, household_id)
    assert count == 2


def test_control_loop_with_langgraph(store, household_id, member_id):
    """Control loop with langgraph_enabled passes the graph phase."""
    from dacribagents.application.services.control_loop import run_cycle

    get_governance_state().set_tier(household_id, AutonomyTier.TIER_1)

    create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Loop+Graph",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )

    result = run_cycle(store, {}, household_id, langgraph_enabled=True)
    assert result.graph_processed >= 1


def test_control_loop_without_langgraph(store, household_id, member_id):
    """Control loop without langgraph_enabled skips the graph phase."""
    from dacribagents.application.services.control_loop import run_cycle

    create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="No graph",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )

    result = run_cycle(store, {}, household_id, langgraph_enabled=False)
    assert result.graph_processed == 0

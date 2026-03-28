"""Tests for the deterministic approval policy."""

from __future__ import annotations

import pytest

from dacribagents.application.services.approval_policy import evaluate
from dacribagents.domain.reminders.enums import ReminderSource, UrgencyLevel


@pytest.mark.parametrize("source", [ReminderSource.USER, ReminderSource.HOUSEHOLD_ROUTINE])
def test_safe_sources_no_approval(source):
    decision = evaluate(source=source)
    assert decision.required is False
    assert decision.reason is None


@pytest.mark.parametrize(
    "source",
    [
        ReminderSource.AGENT,
        ReminderSource.CALENDAR,
        ReminderSource.EMAIL,
        ReminderSource.MAINTENANCE,
        ReminderSource.HOA,
        ReminderSource.WARRANTY,
        ReminderSource.TELEMETRY,
        ReminderSource.EVENT_BUS,
    ],
)
def test_non_safe_sources_require_approval(source):
    decision = evaluate(source=source)
    assert decision.required is True
    assert decision.reason is not None
    assert source.value in decision.reason


def test_override_forces_approval():
    decision = evaluate(source=ReminderSource.USER, requires_approval_override=True)
    assert decision.required is True
    assert "Explicitly" in decision.reason


def test_agent_includes_agent_name():
    decision = evaluate(source=ReminderSource.AGENT, source_agent="hoa_compliance")
    assert decision.required is True
    assert "hoa_compliance" in decision.reason

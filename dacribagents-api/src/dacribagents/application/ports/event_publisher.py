"""Port for publishing domain events to an event backbone.

In production, the implementation publishes to **Kafka** (the Woodcreek
event bus).  LangGraph is the orchestration layer, not the event bus.

This port is a seam — Phase 2C uses a no-op implementation.  Phase 4+
wires to the ``LatticeKafkaPublisher`` or equivalent Kafka producer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class DomainEvent:
    """A domain event ready for publication to the event bus."""

    event_type: str
    reminder_id: UUID
    household_id: UUID
    timestamp: datetime
    payload: dict = field(default_factory=dict)


class EventPublisher(Protocol):
    """Publish domain events to the event backbone (Kafka)."""

    def publish(self, event: DomainEvent) -> None:  # noqa: D102
        ...


class NoOpEventPublisher:
    """Default no-op publisher for phases before Kafka wiring."""

    def publish(self, event: DomainEvent) -> None:  # noqa: D102
        pass

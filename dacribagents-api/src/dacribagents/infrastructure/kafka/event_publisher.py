"""Kafka-backed EventPublisher — publishes reminder domain events to Kafka.

Implements the ``EventPublisher`` protocol from
``application/ports/event_publisher.py``, wiring domain events to the
Kafka event backbone.

Uses ``confluent_kafka.Producer`` with the same SASL_SSL configuration
as the existing ``LatticeKafkaPublisher``.
"""

from __future__ import annotations

import json

from loguru import logger

from dacribagents.application.ports.event_publisher import DomainEvent
from dacribagents.infrastructure.settings import Settings, get_settings


class KafkaEventPublisher:
    """Publishes DomainEvent instances to a Kafka topic."""

    def __init__(
        self,
        settings: Settings | None = None,
        topic: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.topic = topic or self._settings.kafka_topic_events
        self._producer = None

    def _get_producer(self):
        if self._producer is None:
            from confluent_kafka import Producer  # noqa: PLC0415

            config = self._settings.get_kafka_config()
            config["batch.size"] = 65536
            config["linger.ms"] = 50
            self._producer = Producer(config)
            logger.info(f"Kafka event publisher initialized for topic {self.topic}")
        return self._producer

    def publish(self, event: DomainEvent) -> None:
        """Publish a domain event to Kafka."""
        producer = self._get_producer()

        payload = {
            "event_type": event.event_type,
            "reminder_id": str(event.reminder_id),
            "household_id": str(event.household_id),
            "timestamp": event.timestamp.isoformat(),
            "payload": event.payload,
        }

        try:
            producer.produce(
                topic=self.topic,
                key=str(event.reminder_id).encode(),
                value=json.dumps(payload).encode(),
            )
            producer.flush(timeout=5)
            logger.debug(f"Published {event.event_type} for reminder {event.reminder_id}")
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_type}: {e}")

    def close(self) -> None:
        """Flush and close the producer."""
        if self._producer:
            self._producer.flush(timeout=10)
            self._producer = None

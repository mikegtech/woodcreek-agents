"""Kafka consumer for upstream reminder events.

Subscribes to ``woodcreek.events.v1`` (configurable), parses normalized
event payloads, and routes them through the event-intake application
service.

Failed messages are published to a dead-letter topic (DLQ) for
investigation rather than silently dropped.

Aligned with the existing ``LatticeKafkaPublisher`` pattern: SASL_SSL,
confluent_kafka, environment-driven configuration.
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

from loguru import logger

from dacribagents.application.ports.event_publisher import EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services.event_intake import IntakeResult, UpstreamEvent, process_upstream_event
from dacribagents.infrastructure.settings import Settings, get_settings


class ReminderEventConsumer:
    """Kafka consumer that maps upstream events to reminders."""

    def __init__(
        self,
        store: ReminderStore,
        events: EventPublisher | None = None,
        settings: Settings | None = None,
        topic: str | None = None,
    ) -> None:
        self.store = store
        self.events = events or NoOpEventPublisher()
        self._settings = settings or get_settings()
        self.topic = topic or self._settings.kafka_topic_events
        self.dlq_topic = f"{self.topic}.dlq"
        self._consumer = None
        self._dlq_producer = None

    def _get_consumer(self):
        if self._consumer is None:
            from confluent_kafka import Consumer  # noqa: PLC0415

            config = self._settings.get_kafka_config()
            config["group.id"] = self._settings.kafka_consumer_group
            config["auto.offset.reset"] = "earliest"
            config["enable.auto.commit"] = True

            self._consumer = Consumer(config)
            self._consumer.subscribe([self.topic])
            logger.info(f"Kafka consumer subscribed to {self.topic} (group={config['group.id']})")
        return self._consumer

    def _get_dlq_producer(self):
        if self._dlq_producer is None:
            try:
                from confluent_kafka import Producer  # noqa: PLC0415

                config = self._settings.get_kafka_config()
                self._dlq_producer = Producer(config)
            except Exception as e:
                logger.warning(f"DLQ producer init failed (messages will be logged only): {e}")
        return self._dlq_producer

    def run(self, max_messages: int = 100, timeout: float = 1.0) -> list[IntakeResult]:
        """Poll up to *max_messages* and process them. Returns intake results."""
        consumer = self._get_consumer()
        results: list[IntakeResult] = []

        for _ in range(max_messages):
            msg = consumer.poll(timeout=timeout)
            if msg is None:
                break
            if msg.error():
                logger.error(f"Kafka consumer error: {msg.error()}")
                continue

            result = self._process_message(msg.value())
            if result:
                results.append(result)

        return results

    def _process_message(self, raw: bytes) -> IntakeResult | None:
        """Parse and process a single Kafka message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in Kafka message — sending to DLQ")
            self._send_to_dlq(raw, "invalid_json")
            return None

        try:
            event = UpstreamEvent(
                event_type=data["event_type"],
                event_id=data["event_id"],
                household_id=UUID(data["household_id"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                subject=data.get("subject", data.get("event_type", "Untitled")),
                body=data.get("body", ""),
                severity=data.get("severity", "normal"),
                source_service=data.get("source_service", ""),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid event payload: {e} — sending to DLQ")
            self._send_to_dlq(raw, f"invalid_payload: {e}")
            return None

        return process_upstream_event(self.store, event, self.events)

    def _send_to_dlq(self, raw: bytes, reason: str) -> None:
        """Publish a failed message to the dead-letter topic."""
        producer = self._get_dlq_producer()
        if producer is None:
            logger.warning(f"DLQ unavailable — dropped message (reason: {reason})")
            return

        headers = [("dlq_reason", reason.encode())]
        try:
            producer.produce(topic=self.dlq_topic, value=raw, headers=headers)
            producer.flush(timeout=5)
            logger.info(f"Message sent to DLQ {self.dlq_topic}: {reason}")
        except Exception as e:
            logger.error(f"DLQ publish failed: {e}")

    def close(self) -> None:
        """Close the consumer and DLQ producer."""
        if self._consumer:
            self._consumer.close()
            self._consumer = None
        if self._dlq_producer:
            self._dlq_producer.flush(timeout=5)
            self._dlq_producer = None

"""Reminder runtime container — wires production vs dev store/adapters.

Provides a singleton runtime that selects the appropriate store,
calendar adapter, event publisher, and delivery adapters based on
environment configuration.

Usage::

    from dacribagents.infrastructure.reminders.runtime import get_runtime

    rt = get_runtime()
    store = rt.store
    calendar = rt.calendar
"""

from __future__ import annotations

from uuid import UUID

from loguru import logger

from dacribagents.application.ports.event_publisher import EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.infrastructure.settings import get_settings

_DEFAULT_HOUSEHOLD_ID = UUID("00000000-0000-0000-0000-000000000001")


class ReminderRuntime:
    """Central runtime container for reminder subsystem dependencies."""

    def __init__(self) -> None:
        self.store: ReminderStore | None = None
        self.calendar: object | None = None
        self.event_publisher: EventPublisher = NoOpEventPublisher()
        self.adapters: dict = {}
        self.household_id: UUID = _DEFAULT_HOUSEHOLD_ID
        self._scheduler_task = None

    def initialize(self) -> None:
        """Wire all components based on current settings."""
        settings = get_settings()

        # Store selection: Postgres if available, else in-memory
        if settings.environment in {"staging", "production"} or self._postgres_available():
            self.store = self._create_postgres_store()
        else:
            self.store = self._create_memory_store()

        # Calendar: mock for now (WorkMail EWS adapter available but needs credentials)
        from dacribagents.infrastructure.calendar.mock_adapter import MockCalendarAdapter  # noqa: PLC0415

        self.calendar = MockCalendarAdapter()

        # Event publisher: Kafka if configured, else no-op
        if settings.kafka_enabled:
            self.event_publisher = self._create_kafka_publisher()
        else:
            self.event_publisher = NoOpEventPublisher()
            logger.info("Kafka not configured — using NoOpEventPublisher")

        # Delivery adapters
        self.adapters = self._create_delivery_adapters(settings)

        # Load governance state from PostgreSQL if available
        if settings.environment in {"staging", "production"} or self._postgres_available():
            self._load_governance_state()

        logger.info(
            f"Reminder runtime initialized: store={type(self.store).__name__}, "
            f"publisher={type(self.event_publisher).__name__}, "
            f"adapters={list(self.adapters.keys())}"
        )

    def _postgres_available(self) -> bool:
        try:
            from dacribagents.infrastructure import get_postgres_client  # noqa: PLC0415

            client = get_postgres_client()
            client.connect()
            return True
        except Exception:
            return False

    def _create_postgres_store(self) -> ReminderStore:
        from dacribagents.infrastructure import get_postgres_client  # noqa: PLC0415
        from dacribagents.infrastructure.reminders.postgres_store import PostgresReminderStore  # noqa: PLC0415
        from dacribagents.infrastructure.reminders.schema import setup_reminder_schema  # noqa: PLC0415

        client = get_postgres_client()
        conn = client.connect()
        setup_reminder_schema(conn)
        logger.info("PostgresReminderStore initialized with schema")
        return PostgresReminderStore(conn)

    def _create_memory_store(self) -> ReminderStore:
        from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore  # noqa: PLC0415

        logger.info("Using InMemoryReminderStore (dev mode)")
        return InMemoryReminderStore()

    def _create_delivery_adapters(self, settings) -> dict:
        """Register available delivery channel adapters."""
        from dacribagents.domain.reminders.enums import DeliveryChannel  # noqa: PLC0415

        adapters = {}

        # Telnyx SMS
        try:
            from dacribagents.infrastructure.delivery.sms_adapter import TelnyxSmsAdapter  # noqa: PLC0415

            adapters[DeliveryChannel.SMS] = TelnyxSmsAdapter()
            logger.info("Telnyx SMS adapter registered")
        except Exception as e:
            logger.warning(f"Telnyx SMS adapter not available: {e}")

        # Email (SMTP)
        if settings.smtp_user:
            try:
                from dacribagents.infrastructure.delivery.email_adapter import SmtpEmailAdapter  # noqa: PLC0415

                adapters[DeliveryChannel.EMAIL] = SmtpEmailAdapter(
                    smtp_host=settings.smtp_host,
                    smtp_port=settings.smtp_port,
                    smtp_user=settings.smtp_user,
                    smtp_password=settings.smtp_password.get_secret_value() if settings.smtp_password else "",
                    from_address=settings.smtp_from_address,
                )
                logger.info("SMTP email adapter registered")
            except Exception as e:
                logger.warning(f"SMTP email adapter not available: {e}")

        # Slack
        if settings.slack_bot_token:
            try:
                from dacribagents.infrastructure.delivery.slack_adapter import SlackDeliveryAdapter  # noqa: PLC0415

                adapters[DeliveryChannel.SLACK] = SlackDeliveryAdapter(
                    default_channel=settings.slack_delivery_channel,
                )
                logger.info("Slack delivery adapter registered")
            except Exception as e:
                logger.warning(f"Slack delivery adapter not available: {e}")

        return adapters

    def _load_governance_state(self) -> None:
        """Load governance tiers/kill-switch/mutes from PostgreSQL into the in-memory singleton."""
        try:
            from dacribagents.application.services.governance import get_governance_state  # noqa: PLC0415
            from dacribagents.infrastructure import get_postgres_client  # noqa: PLC0415
            from dacribagents.infrastructure.reminders.governance_store import PostgresGovernanceStore  # noqa: PLC0415

            conn = get_postgres_client().connection
            gov_store = PostgresGovernanceStore(conn)
            state = get_governance_state()
            gov_store.load_state(state, self.household_id)
            logger.info(f"Governance state loaded from PostgreSQL: tier={state.get_tier(self.household_id).value}")
        except Exception as e:
            logger.warning(f"Governance state load failed (using defaults): {e}")

    def _create_kafka_publisher(self) -> EventPublisher:
        try:
            from dacribagents.infrastructure.kafka.event_publisher import KafkaEventPublisher  # noqa: PLC0415

            publisher = KafkaEventPublisher()
            logger.info("KafkaEventPublisher initialized")
            return publisher
        except Exception as e:
            logger.warning(f"Kafka publisher init failed, falling back to NoOp: {e}")
            return NoOpEventPublisher()


_runtime: ReminderRuntime | None = None


def get_runtime() -> ReminderRuntime:
    """Get the singleton reminder runtime. Call initialize() during app startup."""
    global _runtime  # noqa: PLW0603
    if _runtime is None:
        _runtime = ReminderRuntime()
    return _runtime


def reset_runtime() -> None:
    """Reset for testing."""
    global _runtime  # noqa: PLW0603
    _runtime = None

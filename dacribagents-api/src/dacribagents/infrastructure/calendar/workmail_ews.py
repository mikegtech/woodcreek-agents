"""AWS WorkMail / EWS read-only calendar adapter.

Implements the ``CalendarAdapter`` protocol using ``exchangelib`` to
connect to AWS WorkMail's EWS endpoint.

WorkMail EWS endpoint pattern:
``https://ews.mail.{region}.awsapps.com/EWS/Exchange.asmx``

This adapter is **read-only** in this phase.  Write operations
(create/delete event) will be added behind ``CalendarAccessPolicy``
in a later phase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from loguru import logger

from dacribagents.application.ports.calendar_adapter import (
    CalendarEvent,
    DateRange,
)
from dacribagents.domain.reminders.enums import CalendarProviderType


class WorkMailEwsAdapter:
    """Read-only calendar adapter backed by AWS WorkMail EWS."""

    def __init__(
        self,
        email: str,
        password: str,
        region: str = "us-east-1",
    ) -> None:
        self._email = email
        self._password = password
        self._region = region
        self._account = None

    @property
    def provider_type(self) -> CalendarProviderType:  # noqa: D102
        return CalendarProviderType.WORKMAIL_EWS

    def _get_account(self):
        """Lazy-initialize the EWS account connection."""
        if self._account is None:
            from exchangelib import (  # noqa: PLC0415
                Account,
                Configuration,
                Credentials,
            )

            server = f"ews.mail.{self._region}.awsapps.com"
            credentials = Credentials(username=self._email, password=self._password)
            config = Configuration(server=server, credentials=credentials)
            self._account = Account(
                primary_smtp_address=self._email,
                config=config,
                autodiscover=False,
                access_type="delegate",
            )
            logger.info(f"WorkMail EWS connected: {self._email}")
        return self._account

    def list_events(
        self,
        identity_id: UUID,
        date_range: DateRange,
    ) -> list[CalendarEvent]:
        """Return calendar items within *date_range*."""
        try:
            account = self._get_account()
            from exchangelib import EWSDateTime, EWSTimeZone  # noqa: PLC0415

            tz = EWSTimeZone("America/Chicago")
            start = EWSDateTime.from_datetime(date_range.start.replace(tzinfo=timezone.utc)).astimezone(tz)
            end = EWSDateTime.from_datetime(date_range.end.replace(tzinfo=timezone.utc)).astimezone(tz)

            items = account.calendar.view(start=start, end=end)
            return [_normalize_event(item, identity_id) for item in items]

        except Exception as e:
            logger.error(f"WorkMail EWS list_events failed: {e}")
            return []

    def get_event(
        self,
        identity_id: UUID,
        event_id: str,
    ) -> CalendarEvent | None:
        """Return a single calendar item by its EWS item_id."""
        try:
            account = self._get_account()
            from exchangelib import ItemId  # noqa: PLC0415

            items = account.calendar.filter(item_id=ItemId(id=event_id))
            for item in items:
                return _normalize_event(item, identity_id)
            return None
        except Exception as e:
            logger.error(f"WorkMail EWS get_event failed: {e}")
            return None

    def list_all_events(self, date_range: DateRange) -> list[CalendarEvent]:
        """Household-wide view — same as list_events for a single-account setup."""
        return self.list_events(UUID(int=0), date_range)

    # ── Write operations (Phase 8C) ────────────────────────────────────

    def create_event(  # noqa: PLR0913
        self,
        identity_id: UUID,
        title: str,
        start: datetime,
        end: datetime,
        body: str = "",
        metadata: dict | None = None,
    ) -> str | None:
        """Create a calendar event. Returns the EWS item_id."""
        try:
            account = self._get_account()
            from exchangelib import CalendarItem, EWSDateTime, EWSTimeZone, HTMLBody  # noqa: PLC0415

            tz = EWSTimeZone("America/Chicago")
            item = CalendarItem(
                account=account,
                folder=account.calendar,
                subject=title,
                body=HTMLBody(body) if body else None,
                start=EWSDateTime.from_datetime(start.replace(tzinfo=timezone.utc)).astimezone(tz),
                end=EWSDateTime.from_datetime(end.replace(tzinfo=timezone.utc)).astimezone(tz),
            )
            item.save()
            event_id = str(item.item_id) if hasattr(item, "item_id") else str(item.id)
            logger.info(f"WorkMail EWS event created: {event_id}")
            return event_id
        except Exception as e:
            logger.error(f"WorkMail EWS create_event failed: {e}")
            return None

    def delete_event(
        self,
        identity_id: UUID,
        event_id: str,
    ) -> bool:
        """Delete a calendar event by EWS item_id."""
        try:
            account = self._get_account()
            from exchangelib import ItemId  # noqa: PLC0415

            items = list(account.calendar.filter(item_id=ItemId(id=event_id)))
            for item in items:
                item.delete()
            logger.info(f"WorkMail EWS event deleted: {event_id}")
            return True
        except Exception as e:
            logger.error(f"WorkMail EWS delete_event failed: {e}")
            return False


def _normalize_event(item: object, identity_id: UUID) -> CalendarEvent:
    """Convert an exchangelib CalendarItem to a provider-neutral CalendarEvent."""
    start = item.start.astimezone(timezone.utc) if item.start else datetime.min.replace(tzinfo=timezone.utc)
    end = item.end.astimezone(timezone.utc) if item.end else start

    return CalendarEvent(
        id=str(getattr(item, "item_id", "") or getattr(item, "id", "")),
        title=item.subject or "(No subject)",
        start=start,
        end=end,
        source=CalendarProviderType.WORKMAIL_EWS,
        owner_identity_id=identity_id,
        description=item.body or "" if hasattr(item, "body") else "",
        all_day=getattr(item, "is_all_day", False) or False,
        recurrence_rule=str(item.recurrence) if getattr(item, "recurrence", None) else None,
        metadata={"location": getattr(item, "location", "") or ""},
    )

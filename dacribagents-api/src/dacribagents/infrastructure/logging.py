"""Structured logging with correlation IDs for reminder operations.

Provides a context-manager that binds a correlation ID (and optional
reminder/household context) to all log messages within its scope.

Usage::

    with reminder_log_context(reminder_id=reminder.id, household_id=hid):
        logger.info("Dispatching reminder")  # includes correlation_id in extra

Also provides ``get_correlation_id()`` for passing to downstream services.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from uuid import UUID, uuid4

from loguru import logger

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Return the current correlation ID (empty string if not set)."""
    return _correlation_id.get()


@contextmanager
def reminder_log_context(
    *,
    reminder_id: UUID | None = None,
    household_id: UUID | None = None,
    operation: str = "",
):
    """Bind structured context to log messages within this scope."""
    cid = uuid4().hex[:12]
    token = _correlation_id.set(cid)
    ctx = {
        "correlation_id": cid,
        "reminder_id": str(reminder_id)[:8] if reminder_id else None,
        "household_id": str(household_id)[:8] if household_id else None,
        "operation": operation,
    }
    with logger.contextualize(**{k: v for k, v in ctx.items() if v}):
        try:
            yield cid
        finally:
            _correlation_id.reset(token)


def configure_structured_logging() -> None:
    """Configure loguru for structured output with correlation IDs.

    Call once during application startup.
    """
    # Add a default empty correlation_id so contextualize works
    logger.configure(extra={"correlation_id": ""})

"""Background scheduler task — runs the reminder control loop periodically.

Started during FastAPI lifespan, cancelled on shutdown.
Runs ``run_cycle()`` every ``SCHEDULER_INTERVAL_SECONDS``.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from dacribagents.infrastructure.logging import reminder_log_context

SCHEDULER_INTERVAL_SECONDS = 60


async def run_scheduler_loop() -> None:
    """Run the reminder control loop indefinitely at a fixed interval."""
    logger.info(f"Scheduler task started (interval={SCHEDULER_INTERVAL_SECONDS}s)")

    while True:
        try:
            await asyncio.sleep(SCHEDULER_INTERVAL_SECONDS)
            with reminder_log_context(operation="scheduler_tick"):
                _run_tick()
        except asyncio.CancelledError:
            logger.info("Scheduler task cancelled (shutdown)")
            break
        except Exception as e:
            logger.error(f"Scheduler tick failed: {e}")
            # Don't crash the loop on transient errors
            await asyncio.sleep(5)


def _run_tick() -> None:
    """Execute one control-loop cycle using the runtime container."""
    from dacribagents.application.services.control_loop import run_cycle  # noqa: PLC0415
    from dacribagents.infrastructure.reminders.runtime import get_runtime  # noqa: PLC0415

    rt = get_runtime()
    if rt.store is None:
        logger.debug("Scheduler tick skipped: runtime not initialized")
        return

    from dacribagents.infrastructure.settings import get_settings  # noqa: PLC0415

    settings = get_settings()
    result = run_cycle(
        rt.store,
        rt.adapters,
        rt.household_id,
        events=rt.event_publisher,
        langgraph_enabled=settings.langgraph_enabled,
    )
    if result.scheduled or result.dispatched or result.retried or result.escalated or result.digested:
        logger.info(
            f"Scheduler tick: scheduled={result.scheduled} dispatched={result.dispatched} "
            f"retried={result.retried} escalated={result.escalated} digested={result.digested}"
        )

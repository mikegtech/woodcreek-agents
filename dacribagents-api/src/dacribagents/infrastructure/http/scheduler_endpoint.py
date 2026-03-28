"""Internal scheduler endpoint — drives the reminder control loop.

Call ``POST /internal/scheduler/tick`` on a regular interval (e.g., every
60 seconds via cron, systemd timer, or FastAPI startup background task).
"""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

router = APIRouter()


@router.post("/internal/scheduler/tick")
async def scheduler_tick() -> dict:
    """Run one control-loop cycle (schedule → dispatch → retry → escalate)."""
    from dacribagents.application.services.control_loop import run_cycle  # noqa: PLC0415
    from dacribagents.infrastructure.slack._store import get_household_store  # noqa: PLC0415

    store, _, household_id = get_household_store()

    # No real adapters in dev — use empty dict (deliveries will record "no adapter" failures)
    result = run_cycle(store, adapters={}, household_id=household_id)

    logger.info(f"Scheduler tick: {result}")
    return {
        "status": "ok",
        "scheduled": result.scheduled,
        "dispatched": result.dispatched,
        "retried": result.retried,
        "escalated": result.escalated,
    }


@router.get("/internal/scheduler/health")
async def scheduler_health() -> dict:
    """Health check for the scheduler subsystem."""
    return {"status": "healthy"}

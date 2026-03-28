"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from dacribagents.infrastructure import (
    get_milvus_client,
    get_postgres_client,
    get_settings,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Configure structured logging
    from dacribagents.infrastructure.logging import configure_structured_logging  # noqa: PLC0415

    configure_structured_logging()

    # Initialize database connections
    milvus = get_milvus_client()
    postgres = get_postgres_client()

    try:
        milvus.connect()
        logger.info("Milvus connection established")
    except Exception as e:
        logger.warning(f"Milvus connection failed (non-fatal): {e}")

    try:
        postgres.connect()
        postgres.setup_schema()
        logger.info("PostgreSQL connection established and schema ready")
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed (non-fatal): {e}")

    # Initialize reminder runtime (selects Postgres or in-memory store)
    from dacribagents.infrastructure.reminders.runtime import get_runtime  # noqa: PLC0415

    runtime = get_runtime()
    try:
        runtime.initialize()
        logger.info("Reminder runtime initialized")
    except Exception as e:
        logger.warning(f"Reminder runtime init failed (non-fatal): {e}")

    # Start scheduler background task
    scheduler_task = None
    if settings.environment != "development" or settings.debug:
        from dacribagents.infrastructure.reminders.scheduler_task import run_scheduler_loop  # noqa: PLC0415

        scheduler_task = asyncio.create_task(run_scheduler_loop())
        logger.info("Scheduler background task started")
    else:
        logger.info("Scheduler background task skipped (development mode — use POST /internal/scheduler/tick)")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")
    if scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass
    milvus.disconnect()
    postgres.disconnect()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Multi-agent home management system for Woodcreek",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from dacribagents.api.routes import router  # noqa: PLC0415
    from dacribagents.infrastructure.agentic_rag.milvus_routes import router as milvus_router  # noqa: PLC0415
    from dacribagents.infrastructure.agentic_rag.routes_combined import router as combined_router  # noqa: PLC0415
    from dacribagents.infrastructure.http.event_ingest import router as event_ingest_router  # noqa: PLC0415
    from dacribagents.infrastructure.http.health import router as subsystem_health_router  # noqa: PLC0415
    from dacribagents.infrastructure.http.scheduler_endpoint import router as scheduler_router  # noqa: PLC0415
    from dacribagents.infrastructure.http.search import router as search_router  # noqa: PLC0415
    from dacribagents.infrastructure.http.slack_ingest import router as slack_router  # noqa: PLC0415
    from dacribagents.infrastructure.http.slack_interactions import (  # noqa: PLC0415
        router as slack_interactions_router,
    )
    from dacribagents.infrastructure.http.sms_ingest import router as sms_router  # noqa: PLC0415
    from dacribagents.infrastructure.ragas import evaluation_router  # noqa: PLC0415

    app.include_router(router)
    app.include_router(sms_router)
    app.include_router(slack_router)
    app.include_router(slack_interactions_router)
    app.include_router(scheduler_router)
    app.include_router(event_ingest_router)
    app.include_router(subsystem_health_router)
    app.include_router(search_router)
    app.include_router(combined_router)
    app.include_router(milvus_router)
    app.include_router(evaluation_router)

    return app


# Create app instance
app = create_app()

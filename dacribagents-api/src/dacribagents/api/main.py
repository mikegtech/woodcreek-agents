"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

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

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")
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
    from dacribagents.api.routes import router
    from dacribagents.infrastructure.http.sms_ingest import router as sms_router

    app.include_router(router)
    app.include_router(sms_router)

    return app


# Create app instance
app = create_app()
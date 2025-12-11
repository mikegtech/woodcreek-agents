"""PostgreSQL client for LangGraph checkpoint persistence."""

from contextlib import asynccontextmanager
from typing import Any

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from loguru import logger

from dacribagents.infrastructure.settings import Settings, get_settings


class PostgresClientWrapper:
    """Wrapper for PostgreSQL operations and LangGraph checkpointing."""

    def __init__(self, settings: Settings | None = None):
        """Initialize PostgreSQL client wrapper."""
        self.settings = settings or get_settings()
        self._connection: psycopg.Connection | None = None
        self._checkpointer: PostgresSaver | None = None

    def connect(self) -> psycopg.Connection:
        """Establish connection to PostgreSQL."""
        if self._connection is None or self._connection.closed:
            logger.info(f"Connecting to PostgreSQL at {self.settings.postgres_host}:{self.settings.postgres_port}")
            self._connection = psycopg.connect(self.settings.postgres_dsn)
            logger.info("PostgreSQL connection established")
        return self._connection

    def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        if self._connection is not None and not self._connection.closed:
            self._connection.close()
            self._connection = None
            self._checkpointer = None
            logger.info("PostgreSQL connection closed")

    @property
    def connection(self) -> psycopg.Connection:
        """Get or create PostgreSQL connection."""
        if self._connection is None or self._connection.closed:
            return self.connect()
        return self._connection

    def health_check(self) -> dict[str, Any]:
        """Check PostgreSQL connection health."""
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
            return {
                "status": "healthy",
                "host": self.settings.postgres_host,
                "port": self.settings.postgres_port,
                "database": self.settings.postgres_db,
                "version": version,
            }
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return {
                "status": "unhealthy",
                "host": self.settings.postgres_host,
                "port": self.settings.postgres_port,
                "database": self.settings.postgres_db,
                "error": str(e),
            }

    def get_checkpointer(self) -> PostgresSaver:
        """Get or create LangGraph PostgreSQL checkpointer."""
        if self._checkpointer is None:
            logger.info("Initializing LangGraph PostgreSQL checkpointer")
            self._checkpointer = PostgresSaver(self.connection)
            self._checkpointer.setup()
            logger.info("LangGraph checkpointer initialized")
        return self._checkpointer

    def setup_schema(self) -> None:
        """Set up database schema for the application."""
        conn = self.connect()
        with conn.cursor() as cur:
            # Create application-specific tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255),
                    agent_type VARCHAR(100) NOT NULL,
                    thread_id VARCHAR(255) NOT NULL UNIQUE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'::jsonb
                );

                CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_id ON agent_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent_type ON agent_sessions(agent_type);
                CREATE INDEX IF NOT EXISTS idx_agent_sessions_thread_id ON agent_sessions(thread_id);
            """)
            conn.commit()
            logger.info("Database schema setup complete")


# Singleton instance
_postgres_client: PostgresClientWrapper | None = None


def get_postgres_client() -> PostgresClientWrapper:
    """Get singleton PostgreSQL client instance."""
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresClientWrapper()
    return _postgres_client


@asynccontextmanager
async def postgres_lifespan():
    """Async context manager for PostgreSQL connection lifecycle."""
    client = get_postgres_client()
    client.connect()
    try:
        yield client
    finally:
        client.disconnect()
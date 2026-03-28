"""Alembic environment configuration.

Uses raw SQL migrations (not SQLAlchemy ORM) to match the existing
psycopg-based schema pattern.
"""

from __future__ import annotations

import os

from alembic import context

# Use TEST_POSTGRES_DSN or alembic.ini default
config = context.config
url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL only)."""
    context.configure(url=url, target_metadata=None, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    from sqlalchemy import create_engine  # noqa: PLC0415

    engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

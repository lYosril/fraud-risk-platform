"""Alembic environment for the PostgreSQL prediction schema.

The database URL is resolved from ``fraud_risk.config.Settings`` so there is no second,
committed connection string. Running in offline mode (``alembic upgrade head --sql``) still
works without a database driver connection.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from fraud_risk.config import get_settings
from fraud_risk.db.models import Prediction

config = context.config

if config.config_file_name is not None:
    # A programmatic migration must not disable the application's existing loggers.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# The URL may contain percent-encoded credentials; ConfigParser interprets `%`, so escape it.
config.set_main_option("sqlalchemy.url", get_settings().sqlalchemy_dsn.replace("%", "%%"))

target_metadata = Prediction.metadata


def run_migrations_offline() -> None:
    """Render migrations to SQL without connecting to PostgreSQL."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations through a short-lived PostgreSQL connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

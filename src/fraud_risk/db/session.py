"""Engine and session lifecycle.

Responsibility
    Build the PostgreSQL engine from ``Settings.sqlalchemy_dsn`` and provide a session
    factory. The FastAPI session dependency lives in ``api.dependencies`` because it is an
    HTTP boundary; this module stays framework-free.

Rules
    Importing this module opens no connection. ``create_engine`` is lazy, so application
    import and unit tests remain usable without a PostgreSQL server.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from fraud_risk.config import Settings

DB_ENGINE_STATE_KEY = "db_engine"
DB_SESSION_FACTORY_STATE_KEY = "db_session_factory"

SessionFactory = sessionmaker[Session]


def create_db_engine(settings: Settings) -> Engine:
    """Create a PostgreSQL engine without connecting to it."""
    return create_engine(
        settings.sqlalchemy_dsn,
        pool_pre_ping=True,
    )


def build_session_factory(engine: Engine) -> SessionFactory:
    """Return a session factory with explicit commit/expiry semantics.

    ``expire_on_commit=False`` keeps the persisted record usable after the repository's
    commit without a second query; autoflush is disabled so repository methods decide when
    SQL is emitted.
    """
    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


__all__ = [
    "DB_ENGINE_STATE_KEY",
    "DB_SESSION_FACTORY_STATE_KEY",
    "SessionFactory",
    "build_session_factory",
    "create_db_engine",
]

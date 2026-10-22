"""Database layer (architecture area 6).

Responsibility
    SQLAlchemy 2.x models, engine/session lifecycle, repository operations and Alembic
    migrations.

Status
    The ``predictions`` table, session factory and API write path are implemented. Feedback
    labels and drift reports remain later milestones. Alembic owns the schema; no application
    code calls ``metadata.create_all()``.
"""

from __future__ import annotations

__all__: list[str] = []

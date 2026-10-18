"""Engine and session lifecycle.

Responsibility
    Build the engine from ``Settings.sqlalchemy_dsn`` and provide a session factory plus a
    FastAPI dependency that guarantees a session is closed per request.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M2).
"""

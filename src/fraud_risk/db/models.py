"""ORM models.

Responsibility
    The three tables defined in ``docs/ARCHITECTURE.md`` §5.3:

    ``predictions``        one row per scored transaction (score, decision, thresholds,
                           model version, JSONB explanation, optional feature snapshot).
    ``feedback_labels``    analyst-confirmed outcomes, kept separate because labels arrive
                           late and incompletely.
    ``drift_reports``      persisted drift runs so the dashboard can show trends.

    The schema is owned by Alembic. ``metadata.create_all()`` is only acceptable in tests.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M2).
"""

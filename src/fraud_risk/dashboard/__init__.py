"""Streamlit analytics dashboard (architecture area 7).

Responsibility
    Read-only views over stored predictions and drift reports.

Boundary
    The dashboard consumes the API (or, for aggregate queries, the database) and never
    imports model internals. It must not become a second, divergent implementation of the
    scoring logic.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M7).
"""

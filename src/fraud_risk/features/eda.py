"""Exploratory data analysis helpers.

Responsibility
    Small, reusable, testable summaries used by the notebooks: class balance, amount and
    time distributions, duplicate/overlap checks between splits, correlation screening.

Rules
    - Importable without plotting libraries: return data, let the caller chart it.
    - Never imported by the API, the dashboard or any inference path.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M3).
"""

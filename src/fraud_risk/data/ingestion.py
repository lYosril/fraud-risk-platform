"""Raw dataset ingestion.

Responsibility
    Read an approved local file into a ``pandas.DataFrame`` under an explicit schema, and
    compute a dataset fingerprint (content hash, row count, time range) so that every
    experiment can be traced back to an exact dataset revision.

Rules
    - Read-only with respect to ``data/raw/``; nothing is written back there.
    - No network downloads: the dataset is placed in ``data/raw/`` by the operator.
    - No silent coercion of invalid values — that belongs to ``data.validation``.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M1).
"""

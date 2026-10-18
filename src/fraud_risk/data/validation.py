"""Dataset contract validation.

Responsibility
    Enforce the canonical transaction schema (dtypes, ranges, nullability, allowed values)
    and the leakage guards, raising an actionable error rather than returning bad data.

Notes
    - The label column (``is_fraud``) must never survive validation on an inference path.
    - Phone-shaped validations that "repair" data are forbidden here: a violation is a
      signal, not something to paper over.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M1).
"""

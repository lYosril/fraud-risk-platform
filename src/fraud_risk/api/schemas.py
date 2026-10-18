"""API request and response schemas.

Responsibility
    Pydantic v2 models mirroring the transaction and prediction contracts in
    ``docs/ARCHITECTURE.md`` §5. These are a trust boundary: they are the only place
    untrusted input becomes typed data.

Rules
    - Reject label fields (``is_fraud``) on the inference path.
    - Validate ranges (amount >= 0, score in [0, 1]) rather than clamping silently.
    - Error responses must not echo secrets or full payloads.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M6).
"""

"""Data ingestion and validation (architecture area 1).

Responsibility
    Turn raw files from ``data/raw/`` into a typed, validated ``pandas.DataFrame`` that the
    rest of the pipeline can trust, and refuse anything that violates the declared contract.

Status
    Implemented: the dataset specification and presence/shape check
    (``fraud_risk.data.dataset``), typed loading (``fraud_risk.data.ingestion``), the schema
    contract and summary report (``fraud_risk.data.validation``), typed errors
    (``fraud_risk.data.errors``) and the CLI (``python -m fraud_risk.data``). The dataset
    fingerprint and the inference-side leakage guard remain in M1/M6 — see
    ``docs/ROADMAP.md``.
"""

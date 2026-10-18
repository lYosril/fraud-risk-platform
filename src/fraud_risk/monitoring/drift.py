"""Drift detection.

Responsibility
    Compare a recent window against the training reference distribution and persist the
    result:

    - Population Stability Index per feature (binned against the training distribution).
    - Kolmogorov-Smirnov two-sample statistic and p-value.
    - Model score distribution shift, prediction volume and decision mix.
    - Delayed-label performance (PR-AUC, recall) over joined ``feedback_labels``.

Design note
    These statistics are implemented here rather than delegated to a framework so that the
    thresholds are version-controlled and the code is unit-testable against distributions
    whose PSI is known analytically. Evidently is an optional extra, used to render rich
    HTML reports — not to make the alert decision (architecture decision D8).

Rules
    - A breach is a prompt to investigate, never an automatic retrain trigger.
    - Thresholds come from ``Settings``, not literals inside analysis scripts.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M8).
"""

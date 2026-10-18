"""Model evaluation.

Responsibility
    Produce the evidence used to choose and to promote a model: PR-AUC (primary metric),
    recall/precision at operating points, ROC-AUC, calibration curve and Brier score, the
    confusion matrix at the tuned thresholds, a business-cost sweep, and error analysis.

Rules
    - Every reported number comes from a real run on a real dataset, with the MLflow run id
      recorded alongside it. No illustrative or invented metrics.
    - Accuracy is never a headline metric on this problem.
    - Thresholds are tuned on validation data only and frozen with the artifact.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M4).
"""

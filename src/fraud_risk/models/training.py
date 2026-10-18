"""Model training.

Responsibility
    Split the data, fit candidate estimators, handle class imbalance, calibrate
    probabilities and log every run (params, metrics, artifacts, dataset fingerprint, split
    definition, git revision) to MLflow.

Rules
    - Temporal split first where the data is time-ordered; a shuffled split on time-ordered
      data inflates metrics and is treated as a bug.
    - Imbalance strategy order: ``scale_pos_weight`` / ``class_weight``, then threshold
      tuning, then resampling only if justified.
    - Calibrate before any threshold is chosen (architecture decision D5).
    - This module never loads model artifacts; that is ``models.registry``.

Status
    Scaffolding only. See ``docs/ROADMAP.md`` (M4).
"""
